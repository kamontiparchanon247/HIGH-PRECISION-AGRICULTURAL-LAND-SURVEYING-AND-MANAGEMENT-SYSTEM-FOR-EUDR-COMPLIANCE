import argparse
import csv
from datetime import date
import math
from pathlib import Path
import re
import sys

import numpy as np
import pyproj
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError

# INPUT 
DEMO = True  
INPUT_MODE = "geometry"  
RTK_GEOMETRY = "POINTZ(100.767529947 13.724620405 2)"
RTK_LAT, RTK_LON = None, None  



RTK_H_ELLIPSOID = None
GEOMETRY_Z_IS_ELLIPSOIDAL = False  


SURVEY_DATE = (2026, 8, 17)

SOURCE_FRAME_CONFIRMED_ITRF2020 = False
TARGET_EPOCH = 2008.87  
TARGET_FRAME = "ITRF2005"

# Everest 1830 (1937 Adjustment), a=6377276.345, 1/f=300.8017
OUTPUT_CRS = "EPSG:24047"  
OUTPUT_DECIMALS = 4  


REFERENCE = dict(
    id="X3446131", e=691461.459063, n=1517667.91215, crs="EPSG:24047",
)
REFERENCE_FROM_MASKED_SCREENSHOT = True

DOL_SHIFT_M = (-204.5, -837.9, -294.8)  # WGS84 ไป Indian 1975
DOL_TRANSFORMATION = "DOL / RTSD 2551, geocentric 3-parameter translation"


CONFIRMED = dict(
    same_monument=False,      
    dol_coordinate_set=False,  
    target_epoch=False,        
    wgs84_bridge=False,        
    motion_model=False,      
)

VE, VN = 24.768e-3, -9.352e-3  
USE_PSD = True
EQ_DATE = (2004, 12, 26)
PSD_N = dict(a=-26.6, tau=0.356)  
PSD_E = dict(a=-42.78, tau=0.356)
USE_MYANMAR = True
MYANMAR_EQ_DATE = (2025, 3, 28)
MYANMAR_COSEISMIC = dict(dE_mm=-0.5217, dN_mm=-1.3811)


A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def parse_geometry(value):
   
    match = re.fullmatch(r"\s*POINT\s*Z\s*\(\s*([^()]+)\s*\)\s*", value, re.I)
    if not match:
        raise ValueError("Geometry ต้องเป็น POINTZ(lon lat height) หรือ POINT Z (...)")
    values = match.group(1).split()
    if len(values) != 3:
        raise ValueError("Geometry ต้องมีตัวเลข 3 ค่า: lon lat height")
    lon, lat, height = map(float, values)
    if not all(math.isfinite(v) for v in (lon, lat, height)):
        raise ValueError("Geometry มีค่าที่ไม่ใช่จำนวนจำกัด")
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("Geometry ต้องเป็นพิกัดองศา lon lat; ตรวจลำดับแกน/CRS")
    return lat, lon, height


def read_input():
    geometry_z = None
    if INPUT_MODE == "geometry":
        lat, lon, geometry_z = parse_geometry(RTK_GEOMETRY)
    elif INPUT_MODE == "latlon":
        if RTK_LAT is None or RTK_LON is None:
            raise ValueError("กรอก RTK_LAT และ RTK_LON จากข้อมูลเต็ม")
        lat, lon = float(RTK_LAT), float(RTK_LON)
    else:
        raise ValueError("INPUT_MODE ต้องเป็น geometry หรือ latlon")

    height = RTK_H_ELLIPSOID
    height_assumed = False
    if height is None and geometry_z is not None and GEOMETRY_Z_IS_ELLIPSOIDAL:
        height = geometry_z
    if height is None:
        if not DEMO:
            raise ValueError("ต้องระบุ RTK_H_ELLIPSOID หรือยืนยันว่า Geometry Z เป็น h")
        height = geometry_z if geometry_z is not None else 0.0
        height_assumed = True
    lat, lon, height = float(lat), float(lon), float(height)
    if not all(math.isfinite(v) for v in (lat, lon, height)):
        raise ValueError("พิกัดต้องเป็นจำนวนจำกัด")
    if not (-80 <= lat <= 84 and -180 <= lon <= 180):
        raise ValueError("พิกัดอยู่นอกช่วง lat/lon ที่รองรับ UTM")
    if not DEMO and not SOURCE_FRAME_CONFIRMED_ITRF2020:
        raise ValueError("ยืนยันเฟรม RTK จากผู้ให้บริการก่อนใช้ ITRF2020 -> ITRF2005")
    return lat, lon, height, height_assumed


def llh_to_ecef(lat_deg, lon_deg, h):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    radius = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.array([(radius + h) * np.cos(lat) * np.cos(lon),
                     (radius + h) * np.cos(lat) * np.sin(lon),
                     (radius * (1 - E2) + h) * np.sin(lat)])


def ecef_to_llh(x, y, z):
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(10):
        radius = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - radius
        lat = np.arctan2(z, p * (1 - E2 * radius / (radius + h)))
    radius = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - radius


def decimal_year(y, m, d):
    start, end, cur = date(y, 1, 1), date(y + 1, 1, 1), date(y, m, d)
    return y + (cur - start).days / (end - start).days


def helmert_itrf2020_to_itrf2005(xyz, t):
    dt = t - 2015.0
    translation = np.array([2.7e-3 + 0.3e-3 * dt,
                            0.1e-3 - 0.1e-3 * dt,
                            -1.4e-3 + 0.1e-3 * dt])
    scale = 0.65e-9 + 0.03e-9 * dt
    return xyz + translation + scale * xyz


def enu_to_ecef(east, north, up, lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    rotation = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ])
    return rotation.T @ np.array([east, north, up])


def psd_offset_mm(a, tau, t, t_eq):
    if tau <= 0:
        raise ValueError("PSD tau ต้องมากกว่า 0")
    return 0.0 if t <= t_eq else a * np.log1p((t - t_eq) / tau)


def correction_steps(lat, lon, height, t_from, t_to):
   
    xyz = llh_to_ecef(lat, lon, height)
    steps = [("RTK raw", xyz.copy())]
    xyz = helmert_itrf2020_to_itrf2005(xyz, t_from)
    steps.append(("Frame only", xyz.copy()))
    xyz += enu_to_ecef(VE, VN, 0, lat, lon) * (t_to - t_from)
    steps.append(("+ Epoch", xyz.copy()))
    if USE_PSD:
        eq_epoch = decimal_year(*EQ_DATE)
        delta = [
            (psd_offset_mm(**p, t=t_to, t_eq=eq_epoch)
             - psd_offset_mm(**p, t=t_from, t_eq=eq_epoch)) / 1000
            for p in (PSD_E, PSD_N)
        ]
        xyz += enu_to_ecef(*delta, 0, lat, lon)
        steps.append(("+ Sumatra PSD", xyz.copy()))
    if USE_MYANMAR:
        eq_epoch = decimal_year(*MYANMAR_EQ_DATE)
        
        factor = int(t_to >= eq_epoch) - int(t_from >= eq_epoch)
        xyz += enu_to_ecef(factor * MYANMAR_COSEISMIC["dE_mm"] / 1000,
                           factor * MYANMAR_COSEISMIC["dN_mm"] / 1000,
                           0, lat, lon)
        steps.append(("+ Myanmar step", xyz.copy()))
    return steps


def grid_difference(en, ref_en):
    """Signed residual = computed Indian grid minus published DOL grid (metres)."""
    d_e, d_n = en[0] - ref_en[0], en[1] - ref_en[1]
    return d_e, d_n, math.hypot(d_e, d_n)


def validate_reference(ref):
    if not isinstance(ref, dict) or ref.get("crs") != OUTPUT_CRS:
        raise ValueError("REFERENCE ต้องระบุ crs=EPSG:24047 ให้ตรงกับปลายทาง")
    if "lat" in ref or "lon" in ref:
        raise ValueError("รุ่น Indian ต้องใช้ E/N ของกรมโดยตรง ไม่รับ reference lat/lon")
    if not ref.get("id"):
        raise ValueError("REFERENCE ต้องมี id ของหมุด")
    for key in ("e", "n"):
        if ref.get(key) is None or not math.isfinite(float(ref[key])):
            raise ValueError(f"กรอก REFERENCE {key} เป็นจำนวนจำกัด")
    if not (100000 <= float(ref["e"]) <= 900000 and 0 <= float(ref["n"]) <= 10000000):
        raise ValueError("REFERENCE E/N อยู่นอกช่วง UTM 47N; ตรวจลำดับแกนและหน่วย")


def build_indian_projector():
    return Transformer.from_pipeline(
        "+proj=pipeline "
        "+step +inv +proj=cart +ellps=evrst30 "
        "+step +proj=utm +zone=47 +ellps=evrst30"
    )


def xyz_to_indian_grid(xyz, projector):
    
    indian_xyz = np.asarray(xyz, dtype=float) + np.asarray(DOL_SHIFT_M, dtype=float)
    east, north, indian_h = projector.transform(*indian_xyz, errcheck=True)
    if not all(math.isfinite(v) for v in (east, north, indian_h)):
        raise ValueError("แปลง Indian 1975 ไม่สำเร็จ")
    return (east, north), indian_xyz, indian_h


def assumption_messages(height_assumed):
    messages = []
    if not SOURCE_FRAME_CONFIRMED_ITRF2020:
        messages.append("สมมติ RTK เป็น ITRF2020 @ วันสำรวจ; ข้อความ WGS84 ในแอปยืนยันไม่ได้")
    if height_assumed:
        messages.append("ใช้ h ชั่วคราวจาก Geometry Z หรือค่าเริ่มต้น; ชนิดความสูงยังไม่ยืนยัน")
    notes = {
        "same_monument": f"สมมติ RTK ตัวอย่างวัดบนหมุด {REFERENCE['id']} เดียวกับ reference",
        "dol_coordinate_set": "ใช้ Indian 1975 และ DOL shift ชุดนี้ตามข้อมูลที่มี; รูปไม่ระบุ datum/ชุดพิกัด",
        "target_epoch": f"ใช้ target epoch {TARGET_EPOCH} ตามโค้ดเดิม; วันนำเข้าระบบในรูปไม่ใช่ epoch",
        "wgs84_bridge": "ใช้ XYZ ใน ITRF เป็น WGS84-equivalent ก่อน DOL shift; ยังไม่มี bridge ที่ยืนยัน",
        "motion_model": "velocity/PSD/Myanmar เป็นแบบจำลองเดิม ยังไม่ได้ตรวจความเหมาะสมกับหมุดนี้",
    }
    messages.extend(note for key, note in notes.items() if not CONFIRMED.get(key, False))
    return messages


def calculate():
    if OUTPUT_CRS != "EPSG:24047" or TARGET_FRAME != "ITRF2005":
        raise ValueError("รุ่นนี้รองรับ ITRF2005 -> Indian 1975 / UTM 47N เท่านั้น")
    if not math.isfinite(TARGET_EPOCH):
        raise ValueError("TARGET_EPOCH ต้องเป็นจำนวนจำกัด")
    if not isinstance(OUTPUT_DECIMALS, int) or not 0 <= OUTPUT_DECIMALS <= 9:
        raise ValueError("OUTPUT_DECIMALS ต้องเป็นจำนวนเต็ม 0-9")
    if tuple(DOL_SHIFT_M) != (-204.5, -837.9, -294.8):
        raise ValueError("DOL_SHIFT_M ต้องตรงกับชุดที่อ้างอิง; การใช้ชุดอื่นต้องเปลี่ยนที่มาและตรวจใหม่")
    validate_reference(REFERENCE)
    lat, lon, height, height_assumed = read_input()
    area = CRS.from_epsg(24047).area_of_use
    if not (area.west <= lon <= area.east and area.south <= lat <= area.north):
        raise ValueError("พิกัดอยู่นอกพื้นที่ใช้งาน Indian 1975 / UTM 47N ของประเทศไทย")
    assumptions = assumption_messages(height_assumed)
    if not DEMO and assumptions:
        raise ValueError("ยังมีข้อมูลที่ไม่ยืนยัน: " + " | ".join(assumptions))
    survey_epoch = decimal_year(*SURVEY_DATE)
    steps = correction_steps(lat, lon, height, survey_epoch, TARGET_EPOCH)
    projector = build_indian_projector()
    ref_en = float(REFERENCE["e"]), float(REFERENCE["n"])
    rows = []
    for index, (label, xyz) in enumerate(steps):
        en, indian_xyz, indian_h = xyz_to_indian_grid(xyz, projector)
        d_e, d_n, distance = grid_difference(en, ref_en)
        rows.append(dict(
            step=label, source_frame="ITRF2020" if index == 0 else TARGET_FRAME,
            source_epoch=survey_epoch if index <= 1 else TARGET_EPOCH,
            e_m=en[0], n_m=en[1], de_m=d_e, dn_m=d_n,
            residual_m=distance, residual_cm=distance * 100,
            x_m=float(xyz[0]), y_m=float(xyz[1]), z_m=float(xyz[2]),
            indian_x_m=float(indian_xyz[0]), indian_y_m=float(indian_xyz[1]),
            indian_z_m=float(indian_xyz[2]), indian_h_m=indian_h,
        ))
    return dict(
        input_lat=lat, input_lon=lon, input_h=height, survey_epoch=survey_epoch,
        height_assumed=height_assumed, assumptions=assumptions, rows=rows,
        reference=dict(REFERENCE), demo=DEMO,
    )


def print_report(result):
    digits = OUTPUT_DECIMALS

    print("RTK -> Indian 1975 / UTM zone 47N (EPSG:24047)")
    print(
        f"Input lat={result['input_lat']:.9f}, "
        f"lon={result['input_lon']:.9f}, "
        f"h={result['input_h']:.4f} m"
    )
    print(
        f"Epoch {result['survey_epoch']:.6f} -> {TARGET_EPOCH}; "
        f"intermediate frame={TARGET_FRAME}"
    )
    

    

    ref = result["reference"]
    print(
        f"Reference {ref['id']} (Indian 1975 / UTM 47N): "
        f"E={ref['e']:.6f}, N={ref['n']:.5f} m"
    )
    

    print("\nแต่ละขั้นฉายเป็น Indian 1975 / UTM 47N ด้วย datum shift ชุดเดียวกัน:")
    for row in result["rows"]:
        print(
            f"{row['step']:18s} "
            f"E={row['e_m']:.{digits}f}  "
            f"N={row['n_m']:.{digits}f}  "
            f"dE={row['de_m']:+.{digits}f}  "
            f"dN={row['dn_m']:+.{digits}f}  "
            f"residual={row['residual_m']:.{digits}f} m "
            f"({row['residual_cm']:.2f} cm)"
        )

    first = result["rows"][0]
    last = result["rows"][-1]

    print(
        f"\nECEF หลังปรับ ({TARGET_FRAME} @ {TARGET_EPOCH}, "
        "ใช้ทรงรี WGS84 ):"
    )
    print(
        f"X={last['x_m']:.{digits}f}, "
        f"Y={last['y_m']:.{digits}f}, "
        f"Z={last['z_m']:.{digits}f} m"
    )

    print("ECEF หลัง DOL datum shift (Indian 1975):")
    print(
        f"X={last['indian_x_m']:.{digits}f}, "
        f"Y={last['indian_y_m']:.{digits}f}, "
        f"Z={last['indian_z_m']:.{digits}f} m"
    )



def write_csv(path, result):
   
    ref = result["reference"]
    metadata = dict(
        reference_id=ref["id"], reference_e_m=ref["e"], reference_n_m=ref["n"],
        output_crs=OUTPUT_CRS, transformation=DOL_TRANSFORMATION,
        shift_x_m=DOL_SHIFT_M[0], shift_y_m=DOL_SHIFT_M[1], shift_z_m=DOL_SHIFT_M[2],
        input_lat=result["input_lat"], input_lon=result["input_lon"], input_h_m=result["input_h"],
        demo=result["demo"], assumptions=" | ".join(result["assumptions"]),
        reference_from_masked_screenshot=REFERENCE_FROM_MASKED_SCREENSHOT,
        interpretation="Single-point grid residual; not certified accuracy; DOL RMS 0.09 m per dimension",
        pyproj_version=pyproj.__version__, proj_version=pyproj.proj_version_str,
    )
    rows = [{**row, **metadata} for row in result["rows"]]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="ไฟล์ผลลัพธ์ทุกขั้นพร้อม residual และสมมติฐาน")
    args = parser.parse_args(argv)
    result = calculate()
    print_report(result)
    if args.csv:
        write_csv(args.csv, result)
        print(f"CSV: {args.csv.resolve()}")
    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, TypeError, ProjError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
