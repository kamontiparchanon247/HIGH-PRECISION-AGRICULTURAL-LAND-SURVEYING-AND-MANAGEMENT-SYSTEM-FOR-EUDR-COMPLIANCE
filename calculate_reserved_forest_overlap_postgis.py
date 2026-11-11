import { forestPool, forestDatasetStateSql } from './forestDatabase';
import proj4 from 'proj4';
import type { ParcelRecord } from '../src/types/geodesy';
import { validatePolygon } from '../src/services/geodesy';
import { ApiError } from './transform';
import { calculateOverlapCentroid } from './forestCentroid';

type Position = [number, number];
type Ring = Position[];
type Polygon = Ring[];
type MultiPolygon = Polygon[];
type Bounds = [number, number, number, number];

function boundsOf(ring: Ring): Bounds {
  return [Math.min(...ring.map(p => p[0])), Math.min(...ring.map(p => p[1])), Math.max(...ring.map(p => p[0])), Math.max(...ring.map(p => p[1]))];
}

export async function checkForestOverlap(
  parcel: ParcelRecord,
  sequence: string[],
) {
  const vertices = new Map(
    parcel.currentVertices.map(vertex => [vertex.id, vertex]),
  );

  if (
    !Array.isArray(sequence)
    || sequence.length < 3
    || sequence.length > 2000
    || sequence.length !== parcel.currentVertices.length
    || new Set(sequence).size !== sequence.length
    || sequence.some(id => !vertices.has(id))
  ) {
    throw new ApiError(422, 'à¸¥à¸³à¸”à¸±à¸šà¸ˆà¸¸à¸”à¸•à¹‰à¸­à¸‡à¸„à¸£à¸šà¸—à¸¸à¸à¸ˆà¸¸à¸” à¹„à¸¡à¹ˆà¸‹à¹‰à¸³ à¹à¸¥à¸°à¹„à¸¡à¹ˆà¹€à¸à¸´à¸™ 2,000 à¸ˆà¸¸à¸”');
  }

  if (!validatePolygon(parcel.currentVertices, sequence).isValid) {
    throw new ApiError(422, 'à¸à¸£à¸¸à¸“à¸²à¹à¸à¹‰à¹„à¸‚à¸‚à¸­à¸šà¹€à¸‚à¸•à¹à¸›à¸¥à¸‡à¸à¹ˆà¸­à¸™à¸•à¸£à¸§à¸ˆà¸›à¹ˆà¸²à¸ªà¸‡à¸§à¸™');
  }

  const ring: Ring = sequence.map(id => {
    const vertex = vertices.get(id)!;
    const hasTransformed = vertex.transformedLat != null
      && vertex.transformedLon != null;
    const lat = hasTransformed
      ? vertex.transformedLat : vertex.latitude;
    const lon = hasTransformed
      ? vertex.transformedLon : vertex.longitude;

    if (
      typeof lat !== 'number' || typeof lon !== 'number'
      || !Number.isFinite(lat) || !Number.isFinite(lon)
      || lon < 90 || lon > 110 || lat < 0 || lat > 25
    ) {
      throw new ApiError(422, 'à¸žà¸´à¸à¸±à¸”à¹à¸›à¸¥à¸‡à¹„à¸¡à¹ˆà¸žà¸£à¹‰à¸­à¸¡à¸•à¸£à¸§à¸ˆà¸›à¹ˆà¸²à¸ªà¸‡à¸§à¸™à¹ƒà¸™à¸›à¸£à¸°à¹€à¸—à¸¨à¹„à¸—à¸¢');
    }

    return [lon, lat];
  });

  const first = ring[0];
  const last = ring[ring.length - 1];
  if (first[0] !== last[0] || first[1] !== last[1]) {
    ring.push([...first]);
  }

  const currentPolygon: Polygon = [ring];
  const parcelBounds = boundsOf(ring);

  // à¹€à¸¥à¸·à¸­à¸ UTM à¸•à¸²à¸¡à¸•à¸³à¹à¸«à¸™à¹ˆà¸‡à¹à¸›à¸¥à¸‡à¸ªà¸³à¸«à¸£à¸±à¸šà¸„à¸³à¸™à¸§à¸“à¸žà¸·à¹‰à¸™à¸—à¸µà¹ˆà¸«à¸™à¹ˆà¸§à¸¢à¹€à¸¡à¸•à¸£
  const centerLon = (parcelBounds[0] + parcelBounds[2]) / 2;
  const zone = Math.floor((centerLon + 180) / 6) + 1;
  const projection = proj4(
    '+proj=longlat +datum=WGS84 +no_defs',
    `+proj=utm +zone=${zone} +datum=WGS84 +units=m +no_defs`,
  );

  function ringArea(points: Ring): number {
    const projected = points.map(p => projection.forward(p));
    if (projected.length < 3) return 0;

    // à¹€à¸¥à¸·à¹ˆà¸­à¸™à¸ˆà¸¸à¸”à¸à¸³à¹€à¸™à¸´à¸”à¹€à¸žà¸·à¹ˆà¸­à¸¥à¸”à¸„à¸§à¸²à¸¡à¸„à¸¥à¸²à¸”à¹€à¸„à¸¥à¸·à¹ˆà¸­à¸™à¸ˆà¸²à¸à¸•à¸±à¸§à¹€à¸¥à¸‚à¸žà¸´à¸à¸±à¸”à¸‚à¸™à¸²à¸”à¹ƒà¸«à¸à¹ˆ
    const [originX, originY] = projected[0];
    let sum = 0;

    for (let i = 0; i < projected.length; i++) {
      const a = projected[i];
      const b = projected[(i + 1) % projected.length];
      sum += (a[0] - originX) * (b[1] - originY)
        - (b[0] - originX) * (a[1] - originY);
    }

    return Math.abs(sum) / 2;
  }

  function area(geometry: MultiPolygon): number {
    return geometry.reduce((total, polygon) => {
      if (!polygon.length) return total;
      const holes = polygon.slice(1).reduce(
        (sum, hole) => sum + ringArea(hole), 0,
      );
      return total + Math.max(0, ringArea(polygon[0]) - holes);
    }, 0);
  }

  const parcelArea = area([currentPolygon]);
  if (!Number.isFinite(parcelArea) || parcelArea <= 0) {
    throw new ApiError(422, 'à¹„à¸¡à¹ˆà¸ªà¸²à¸¡à¸²à¸£à¸–à¸„à¸³à¸™à¸§à¸“à¸žà¸·à¹‰à¸™à¸—à¸µà¹ˆà¹à¸›à¸¥à¸‡à¹„à¸”à¹‰');
  }

  if (!process.env.FOREST_DATABASE_URL) throw new ApiError(503, 'à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸²à¸ªà¸‡à¸§à¸™à¸¢à¸±à¸‡à¹„à¸¡à¹ˆà¸žà¸£à¹‰à¸­à¸¡à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸š');
  let rows;
  try {
    const result = await forestPool.query(`
      WITH dataset_state AS MATERIALIZED (${forestDatasetStateSql}),
      parcel AS (SELECT ST_SetSRID(ST_GeomFromGeoJSON($1), 4326) AS geom),
      intersections AS MATERIALIZED (
        SELECT f.code, f.name, f.province, f.requires_boundary_review, f.geom AS forest_geom,
          ST_Multi(ST_CollectionExtract(ST_Intersection(f.geom, p.geom), 3)) AS overlap
        FROM forest.reserved_forest f CROSS JOIN parcel p CROSS JOIN dataset_state d
        WHERE d.ready AND ST_Intersects(f.geom, p.geom)
      ), matched AS MATERIALIZED (
        SELECT *, ST_Area(ST_Transform(overlap, $2::integer)) AS square_meters FROM intersections
        WHERE NOT ST_IsEmpty(overlap)
      ), selected AS MATERIALIZED (SELECT * FROM matched WHERE square_meters > 0.01),
      combined AS (SELECT ST_Multi(ST_CollectionExtract(ST_UnaryUnion(ST_Collect(overlap)), 3)) AS geom FROM selected)
      SELECT (SELECT ready FROM dataset_state) AS dataset_ready,
      COALESCE((SELECT json_agg(json_build_object(
        'code', code, 'name', name, 'province', province,
        'requiresBoundaryReview', requires_boundary_review,
        'overlapSquareMeters', square_meters,
        'overlapGeometry', ST_AsGeoJSON(overlap, 15)::json->'coordinates',
        'forestGeometry', ST_AsGeoJSON(forest_geom, 15)::json->'coordinates'
      ) ORDER BY square_meters DESC) FROM selected), '[]'::json) AS matches,
      COALESCE((SELECT ST_AsGeoJSON(geom, 15)::json->'coordinates' FROM combined), '[]'::json) AS total_geometry,
      COALESCE((SELECT ST_Area(ST_Transform(geom, $2::integer)) FROM combined), 0) AS total_area
    `, [JSON.stringify({ type: 'Polygon', coordinates: currentPolygon }), 32600 + zone]);
    rows = result.rows[0];
    if (rows?.dataset_ready !== true) {
      throw new ApiError(503, 'à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸²à¸ªà¸‡à¸§à¸™à¸¢à¸±à¸‡à¹„à¸¡à¹ˆà¸žà¸£à¹‰à¸­à¸¡à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸š');
    }
  } catch (error) {
    console.error('Forest PostGIS query failed', { code: (error as { code?: string }).code ?? 'unavailable' });
    throw new ApiError(503, 'à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸²à¸ªà¸‡à¸§à¸™à¸¢à¸±à¸‡à¹„à¸¡à¹ˆà¸žà¸£à¹‰à¸­à¸¡à¸•à¸£à¸§à¸ˆà¸ªà¸­à¸š');
  }
  const matches = rows.matches.map((m: any) => ({ ...m, overlapPercent: Math.min(100, m.overlapSquareMeters / parcelArea * 100) }));
  const totalOverlapGeometry: MultiPolygon = rows.total_geometry;
  const totalOverlapSquareMeters: number = rows.total_area;
  if (!Number.isFinite(totalOverlapSquareMeters)) throw new ApiError(503, 'à¸„à¸³à¸™à¸§à¸“à¸žà¸·à¹‰à¸™à¸—à¸µà¹ˆà¸—à¸±à¸šà¸‹à¹‰à¸­à¸™à¸£à¸§à¸¡à¹„à¸¡à¹ˆà¸ªà¸³à¹€à¸£à¹‡à¸ˆ');

  const totalOverlapPercent = Math.min(
    100,
    totalOverlapSquareMeters / parcelArea * 100,
  );

  const overlapCentroid = calculateOverlapCentroid(
    totalOverlapGeometry,
    zone,
  );


  return {
    parcelId: parcel.id,
    checkedAt: new Date().toISOString(),
    source: 'à¸à¸£à¸¡à¸›à¹ˆà¸²à¹„à¸¡à¹‰',
    sourceUrl: 'https://data.forest.go.th/dataset/reserve_forest',
    datasetFile: 'final_NRF_all_1221_620108',
    datasetBoundaryDate: null,
    screeningOnly: true,
    coordinateNote:
      'à¹€à¸›à¸£à¸µà¸¢à¸šà¹€à¸—à¸µà¸¢à¸šà¸žà¸´à¸à¸±à¸”à¸ªà¸³à¸«à¸£à¸±à¸šà¹à¸ªà¸”à¸‡à¸šà¸™à¹à¸œà¸™à¸—à¸µà¹ˆ à¸¢à¸±à¸‡à¹„à¸¡à¹ˆà¹„à¸”à¹‰à¸›à¸£à¸±à¸šà¸„à¸§à¸²à¸¡à¸•à¹ˆà¸²à¸‡à¸‚à¸­à¸‡à¸à¸£à¸­à¸šà¸­à¹‰à¸²à¸‡à¸­à¸´à¸‡à¹à¸¥à¸° epoch à¸£à¸°à¸«à¸§à¹ˆà¸²à¸‡à¹à¸›à¸¥à¸‡à¸à¸±à¸šà¸Šà¸¸à¸”à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸›à¹ˆà¸²',
    areaMethod: `WGS84 / UTM zone ${zone}N`,
    currentPolygon,
    parcelSquareMeters: parcelArea,
    totalOverlapGeometry,
    totalOverlapSquareMeters,
    totalOverlapPercent,
    overlapCentroid,
    matches,
  };
}
