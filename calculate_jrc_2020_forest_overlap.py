"""Read native GFC2020 V4 classification pixels, never RGB map tiles."""
import math
import os
import secrets
import threading
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape, mapping, box
from shapely.ops import transform, unary_union

app = FastAPI(title="RTK EUDR baseline screening")
SOURCE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFC2020/LATEST/single-cog/JRC_GFC2020_V4_COG.tif"
SOURCE_ETAG = '"96d7eb629-65b6e09115bb3"'
MAX_PIXELS = 250_000
slots = threading.BoundedSemaphore(2)
TOKEN = os.environ.get("EUDR_INTERNAL_TOKEN", "")


class CheckInput(BaseModel):
    geometry: dict


def authorize(token):
    if not TOKEN or not token or not secrets.compare_digest(TOKEN, token):
        raise HTTPException(401, "Unauthorized")


def validate_geometry(data):
    try:
        if data.get("type") != "Polygon":
            raise ValueError()
        rings = data["coordinates"]
        if not isinstance(rings, (list, tuple)) or not rings or len(rings) > 50:
            raise ValueError()
        if sum(len(r) for r in rings) > 2050:
            raise ValueError()
        for ring in rings:
            if len(ring) < 4 or ring[0] != ring[-1]:
                raise ValueError()
            for p in ring:
                if len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p):
                    raise ValueError()
                if not (97 <= p[0] <= 106 and 5 <= p[1] <= 21):
                    raise ValueError()
        parcel = shape(data)
        if parcel.is_empty or not parcel.is_valid or parcel.area <= 0:
            raise ValueError()
        return parcel
    except (ValueError, TypeError, KeyError, IndexError):
        raise HTTPException(422, "à¸‚à¸­à¸šà¹€à¸‚à¸•à¹à¸›à¸¥à¸‡à¹„à¸¡à¹ˆà¸–à¸¹à¸à¸•à¹‰à¸­à¸‡ à¸«à¸£à¸·à¸­à¸­à¸¢à¸¹à¹ˆà¸™à¸­à¸à¸Šà¹ˆà¸§à¸‡à¸žà¸·à¹‰à¸™à¸—à¸µà¹ˆà¸›à¸£à¸°à¹€à¸—à¸¨à¹„à¸—à¸¢à¸—à¸µà¹ˆà¸£à¸­à¸‡à¸£à¸±à¸š")


def calculate_overlap(parcel, data, affine):
    """Clip the full native pixel footprint, including fractional boundary cells."""
    if np.any((data != 0) & (data != 1)):
        raise HTTPException(503, "à¸žà¸šà¸„à¹ˆà¸²à¸ˆà¸³à¹à¸™à¸à¸—à¸µà¹ˆà¹„à¸¡à¹ˆà¸£à¸¹à¹‰à¸ˆà¸±à¸à¹ƒà¸™à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸² à¸à¸£à¸¸à¸“à¸²à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¸Šà¸¸à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥")
    polygons = [shape(g) for g, value in shapes(data, mask=data == 1, transform=affine) if value == 1]
    forest = unary_union(polygons).intersection(parcel) if polygons else parcel.difference(parcel)
    # Remove lines/points from boundary-only contact; they have no forest area.
    if forest.geom_type == "GeometryCollection":
        forest = unary_union([g for g in forest.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
    zone = int((parcel.centroid.x + 180) // 6) + 1
    project = Transformer.from_crs(4326, 32600 + zone, always_xy=True).transform
    parcel_area = transform(project, parcel).area
    overlap_area = transform(project, forest).area if not forest.is_empty else 0.0
    return {
        "parcelSquareMeters": parcel_area,
        "forestSquareMeters": overlap_area,
        "forestPercent": min(100.0, overlap_area / parcel_area * 100),
        "status": "OVERLAP" if overlap_area > 0.01 else "NO_OVERLAP_DETECTED",
        "overlapGeometry": mapping(forest) if not forest.is_empty else None,
        "areaMethod": f"WGS84 / UTM {zone}N; native raster footprint clipped to parcel",
        "windowPixelCount": int(data.size),
        "windowForestPixelCount": int(np.count_nonzero(data == 1)),
    }


def read_and_check(parcel):
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_TIMEOUT="15", GDAL_HTTP_CONNECTTIMEOUT="8",
        GDAL_HTTP_MAX_RETRY="0", GDAL_HTTP_HEADER_FILE=str(Path(__file__).with_name("source.headers")),
        VSI_CACHE=True, VSI_CACHE_SIZE=16777216,
    ):
        with rasterio.open(SOURCE_URL) as ds:
            if ds.crs.to_epsg() != 4326 or ds.count != 1 or ds.dtypes[0] != "uint8" or ds.nodata is not None:
                raise HTTPException(503, "à¹‚à¸„à¸£à¸‡à¸ªà¸£à¹‰à¸²à¸‡à¸Šà¸¸à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸²à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™ à¸•à¹‰à¸­à¸‡à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸šà¸£à¸¸à¹ˆà¸™à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸à¹ˆà¸­à¸™à¹ƒà¸Šà¹‰à¸‡à¸²à¸™")
            if not box(*ds.bounds).covers(parcel):
                raise HTTPException(422, "à¹à¸›à¸¥à¸‡à¸­à¸¢à¸¹à¹ˆà¸™à¸­à¸à¸‚à¸­à¸šà¹€à¸‚à¸•à¸‚à¹‰à¸­à¸¡à¸¹à¸¥")
            w = from_bounds(*parcel.bounds, ds.transform)
            col, row = math.floor(w.col_off), math.floor(w.row_off)
            width, height = math.ceil(w.col_off + w.width) - col, math.ceil(w.row_off + w.height) - row
            if width * height > MAX_PIXELS:
                raise HTTPException(422, "à¸‚à¸­à¸šà¹€à¸‚à¸•à¹à¸›à¸¥à¸‡à¹ƒà¸«à¸à¹ˆà¹€à¸à¸´à¸™à¸à¸²à¸£à¸•à¸£à¸§à¸ˆà¸„à¸£à¸±à¹‰à¸‡à¹€à¸”à¸µà¸¢à¸§ à¸à¸£à¸¸à¸“à¸²à¹à¸šà¹ˆà¸‡à¹€à¸›à¹‡à¸™à¹à¸›à¸¥à¸‡à¸¢à¹ˆà¸­à¸¢")
            window = Window(col, row, width, height)
            data = ds.read(1, window=window)  # Native resolution; no resampling/overview.
            if data.shape != (height, width):
                raise HTTPException(503, "à¸­à¹ˆà¸²à¸™à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸²à¹„à¸”à¹‰à¹„à¸¡à¹ˆà¸„à¸£à¸š")
            return calculate_overlap(parcel, data, ds.window_transform(window))


@app.get("/health")
def health():
    return {"status": "ok", "dataset": "JRC_GFC2020_V4", "tokenConfigured": bool(TOKEN)}


@app.post("/check")
def check(body: CheckInput, x_eudr_token: str | None = Header(default=None)):
    authorize(x_eudr_token)
    parcel = validate_geometry(body.geometry)
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "à¸¡à¸µà¸à¸²à¸£à¸•à¸£à¸§à¸ˆà¸›à¹ˆà¸²à¸žà¸£à¹‰à¸­à¸¡à¸à¸±à¸™ à¸à¸£à¸¸à¸“à¸²à¸¥à¸­à¸‡à¸­à¸µà¸à¸„à¸£à¸±à¹‰à¸‡")
    try:
        result = read_and_check(parcel)
        return {
            **result,
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "dataset": "JRC_GFC2020_V4", "baselineDate": "2020-12-31",
            "source": "European Commission, Joint Research Centre",
            "sourceUrl": "https://forobs.jrc.ec.europa.eu/GFC",
            "sourceRasterUrl": SOURCE_URL, "sourceEtag": SOURCE_ETAG,
            "citation": "Bourgoin, Clement; Achard, Frederic; Beuchle, Rene; Carboni, Silvia; Carreiras, Joao; Marinelli, Daniele; Rotllan-Puig, Xavier; Simonetti, Dario; Colditz, Rene (2026): Global map of forest cover 2020 - version 4. European Commission, Joint Research Centre [Dataset]. doi:10.2905/JRC.3KATEH8",
            "copyright": "Â© European Union, 1995â€“2026",
            "licence": "CC BY 4.0 (notice distributed with the raster download)",
            "licenceNoticeUrl": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFC2020/copyright.txt",
            "changes": "Native forest raster polygonized, clipped to the supplied parcel and measured in UTM; original raster unchanged.",
            "nominalResolutionMeters": 10, "screeningOnly": True,
            "notes": [
                "à¸•à¸£à¸§à¸ˆà¸à¸²à¸£à¸—à¸±à¸šà¸‹à¹‰à¸­à¸™à¸›à¹ˆà¸²à¸à¸²à¸™à¸›à¸µ 2020 à¸¢à¸±à¸‡à¹„à¸¡à¹ˆà¹„à¸”à¹‰à¸•à¸£à¸§à¸ˆà¸à¸²à¸£à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¹à¸›à¸¥à¸‡à¸«à¸¥à¸±à¸‡à¸›à¸µ 2020 à¸«à¸£à¸·à¸­à¸£à¸±à¸šà¸£à¸­à¸‡ EUDR",
                "à¸„à¹ˆà¸²à¸žà¸·à¹‰à¸™à¸—à¸µà¹ˆà¸„à¸³à¸™à¸§à¸“à¸ˆà¸²à¸à¸žà¸´à¸à¹€à¸‹à¸¥à¸ˆà¸³à¹à¸™à¸à¸›à¹ˆà¸² GeoTIFF à¸ˆà¸£à¸´à¸‡ à¹‚à¸”à¸¢à¸•à¸±à¸”à¸ªà¹ˆà¸§à¸™à¸žà¸´à¸à¹€à¸‹à¸¥à¸•à¸²à¸¡à¸‚à¸­à¸šà¹à¸›à¸¥à¸‡",
                "à¸„à¸§à¸²à¸¡à¸¥à¸°à¹€à¸­à¸µà¸¢à¸”à¸›à¸£à¸°à¸¡à¸²à¸“ 10 à¹€à¸¡à¸•à¸£; à¸‚à¸­à¸šà¸›à¹ˆà¸²à¹à¸¥à¸°à¸žà¸´à¸à¸±à¸”à¹à¸›à¸¥à¸‡à¸­à¸²à¸ˆà¸„à¸¥à¸²à¸”à¹€à¸„à¸¥à¸·à¹ˆà¸­à¸™ à¹à¸¥à¸°à¸¢à¸±à¸‡à¹„à¸¡à¹ˆà¹„à¸”à¹‰à¸›à¸£à¸±à¸š frame/epoch à¹ƒà¸«à¹‰à¸•à¸£à¸‡à¸à¸±à¸™",
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"EUDR raster unavailable: {type(exc).__name__}", flush=True)
        raise HTTPException(503, "à¸­à¹ˆà¸²à¸™à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸²à¸ˆà¸²à¸ JRC à¹„à¸¡à¹ˆà¸ªà¸³à¹€à¸£à¹‡à¸ˆ à¸à¸£à¸¸à¸“à¸²à¸¥à¸­à¸‡à¹ƒà¸«à¸¡à¹ˆ; à¸¢à¸±à¸‡à¸ªà¸£à¸¸à¸›à¸œà¸¥à¹à¸›à¸¥à¸‡à¸™à¸µà¹‰à¹„à¸¡à¹ˆà¹„à¸”à¹‰")
    finally:
        slots.release()
