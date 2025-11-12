"""
Snaps POIs from a filtered PBF to the nearest graph (CSR) nodes and writes a Parquet file to data/.

Input:
  --pbf   : filtered .pbf tile (contains POIs + pedestrian network)
  --csr   : CSR .npz file produced by graph_construction.py (…_csr.npz)
  --out   : output Parquet path (default: data/{basename}_poi.parquet)

Output (Parquet):
  poi_id:int64         — OSM POI ID (if missing → negative synthetic ID)
  category:str         — category from the predefined dictionary
  node_idx:int32       — node index in CSR (0..N-1)
  lon:float32, lat:float32
  name:str (may be NaN)
  dist_to_node_m:float32 — distance to snapped node (meters; for debug/QA)

Categories (per your list):
  bus_stop, playground, convenience, school, park, supermarket, parcel_locker,
  kinder_childcare, pharmacy, bakery, clinic_hospital, tram_stop, library,
  university, pub, rail_station, veterinary, fitness_centre, pet_shop, nightclub
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

from pyrosm import OSM
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.errors import GEOSException
from shapely import wkt
from scipy.spatial import cKDTree
from pyproj import Transformer

# ---------- utils ----------
def tsec(t0): return f"{(time.perf_counter()-t0):.3f}s"

def safe_centroid(geom):
    if geom is None:
        return None
    try:
        c = geom.centroid
        if not isinstance(c, Point) or not np.isfinite(c.x) or not np.isfinite(c.y):
            return None
        return c
    except GEOSException:
        try:
            g = wkt.loads(geom) if isinstance(geom, str) else None
            return g.centroid if g is not None else None
        except Exception:
            return None
    except Exception:
        return None

def to_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    mask_point = gdf.geometry.geom_type.eq("Point")
    pts = gdf[mask_point].copy()
    not_pts = gdf[~mask_point].copy()
    if not not_pts.empty:
        not_pts["geometry"] = not_pts.geometry.apply(safe_centroid)
        not_pts = not_pts[not_pts.geometry.notnull()]
    out = pd.concat([pts, not_pts], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)

def load_csr(csr_path: str):
    csr = np.load(csr_path)
    indptr  = csr["indptr"].astype(np.int32, copy=False)
    indices = csr["indices"].astype(np.int32, copy=False)
    weights = csr["weights"].astype(np.float32, copy=False)
    lonlat  = csr["lonlat"].astype(np.float32, copy=False)  # [N,2]
    osm_node_id = csr["osm_node_id"].astype(np.int64, copy=False)
    return indptr, indices, weights, lonlat, osm_node_id

def make_tree_3857(lonlat: np.ndarray):
    transformer = Transformer.from_crs(4326, 3857, always_xy=True)
    x, y = transformer.transform(lonlat[:,0], lonlat[:,1])
    coords = np.column_stack([x.astype(np.float32), y.astype(np.float32)])
    tree = cKDTree(coords)
    return tree, coords, transformer

TAG_MAP: Dict[str, List[Tuple[str, str]]] = {
    "supermarket":       [("shop","supermarket")],
    "convenience":       [("shop","convenience")],
    "bakery":            [("shop","bakery")],
    "pet_shop":          [("shop","pet")],
    
    "pharmacy":          [("amenity","pharmacy")],
    "clinic_hospital":   [("amenity","clinic"), ("amenity","hospital")],
    "parcel_locker":     [("amenity","parcel_locker")],
    "university":        [("amenity","university"), ("amenity","college")],
    "library":           [("amenity","library")],
    "nightclub":         [("amenity","nightclub")],
    "school":            [("amenity","school")],
    "kinder_childcare":  [("amenity","kindergarten"), ("amenity","childcare")],
    "veterinary":        [("amenity","veterinary")],
    "pub":               [("amenity","pub")],

    "fitness_centre":    [("leisure","fitness_centre")],
    "playground":        [("leisure","playground")],
    "park":              [("leisure","park")],
  
    "bus_stop":          [("highway","bus_stop")],
    "tram_stop":         [("railway","tram_stop")],

    "rail_station":      [("railway","station"), ("railway","halt"),
                          ("public_transport","station"), ("public_transport","halt")],
}

OSM_KEYS = sorted({k for pairs in TAG_MAP.values() for (k, v) in pairs})

def extract_pois(osm: OSM) -> gpd.GeoDataFrame:
    pieces = []
    for cat, pairs in TAG_MAP.items():
        custom = {}
        for (key, val) in pairs:
            custom.setdefault(key, []).append(val)

        try:
            gdf = osm.get_data_by_custom_criteria(
                custom_filter=custom,
                filter_type="keep",
                keep_nodes=True,
                keep_ways=True,
                keep_relations=True,  
            )
        except KeyError as e:
            if "tags" in str(e):
                #print("[snap] WARNING: relations without 'tags' in this PBF -> retry without relations")
                gdf = osm.get_data_by_custom_criteria(
                    custom_filter=custom,
                    filter_type="keep",
                    keep_nodes=True,
                    keep_ways=True,
                    keep_relations=False, 
                )
            else:
                raise

        if gdf is None or gdf.empty:
            #print("[snap] INFO: Brak POI po filtrze (nodes/ways/relations).")
            continue

        gdf = to_points(gdf)
        if gdf.empty:
            continue

        if "id" in gdf.columns:
            gdf["poi_id"] = pd.to_numeric(gdf["id"], errors="coerce").astype("Int64")
        elif "osmid" in gdf.columns:
            gdf["poi_id"] = pd.to_numeric(gdf["osmid"], errors="coerce").astype("Int64")
        else:
            gdf["poi_id"] = pd.Series([-i for i in range(1, len(gdf)+1)], dtype="Int64")

        if "name" not in gdf.columns:
            gdf["name"] = None

        gdf["category"] = cat
        pieces.append(gdf[["poi_id","name","category","geometry"]])

    if not pieces:
        return gpd.GeoDataFrame(columns=["poi_id","name","category","geometry"], geometry="geometry", crs="EPSG:4326")

    out = pd.concat(pieces, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    out["poi_id"] = out["poi_id"].astype("int64", errors="ignore")
    return out

def snap_pois_to_nodes(pois_gdf: gpd.GeoDataFrame, lonlat_nodes: np.ndarray, tree_3857, nodes_3857, transformer):
    if pois_gdf.empty:
        return pd.DataFrame(columns=["node_idx","dist_to_node_m"])

    x, y = transformer.transform(pois_gdf.geometry.x.values, pois_gdf.geometry.y.values)
    poi_xy = np.column_stack([x.astype(np.float32), y.astype(np.float32)])

    dists, idxs = tree_3857.query(poi_xy, k=1)
    res = pd.DataFrame({
        "node_idx": idxs.astype(np.int32),
        "dist_to_node_m": dists.astype(np.float32),
    })
    return res

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True, help="Filtered .pbf tile (contains POIs and network).")
    ap.add_argument("--csr", required=True, help="CSR .npz file for the SAME tile.")
    ap.add_argument("--out", default=None, help="Output Parquet path. Defaults to data/{basename}_poi.parquet")
    args = ap.parse_args()

    T0 = time.perf_counter()
    print(f"[0] Start | pbf={args.pbf}  csr={args.csr}")

    t = time.perf_counter()
    indptr, indices, weights, lonlat, osm_node_id = load_csr(args.csr)
    tree, nodes_3857, transformer = make_tree_3857(lonlat)
    N = lonlat.shape[0]
    print(f"[1] CSR: nodes={N}, KDTree built ok  ({tsec(t)})")

    t = time.perf_counter()
    osm = OSM(args.pbf)
    pois = extract_pois(osm)
    if pois.empty:
        print("[2] No POIs in this tile per the filter.")
        out_path = args.out or os.path.join("data", f"{os.path.splitext(os.path.basename(args.pbf))[0]}_poi.parquet")
        pd.DataFrame(columns=["poi_id","category","node_idx","lon","lat","name","dist_to_node_m"]).to_parquet(out_path, index=False)
        print(f"[OK] Saved empty {out_path}.  ({tsec(t)})")
        return
    print(f"[2] POIs extracted: {len(pois)}  ({tsec(t)})")

    pois["lon"] = pois.geometry.x.astype("float32")
    pois["lat"] = pois.geometry.y.astype("float32")

    t = time.perf_counter()
    snapped = snap_pois_to_nodes(pois, lonlat, tree, nodes_3857, transformer)
    print(f"[3] Snapped: {len(snapped)} ({tsec(t)})")

    out_df = pd.DataFrame({
        "poi_id": pois["poi_id"].astype("int64"),
        "category": pois["category"].astype("string"),
        "node_idx": snapped["node_idx"].astype("int32"),
        "lon": pois["lon"].astype("float32"),
        "lat": pois["lat"].astype("float32"),
        "name": pois["name"].astype("string"),
        "dist_to_node_m": snapped["dist_to_node_m"].astype("float32"),
    })

    cwd = os.getcwd()
    if getattr(args, "out", None):
        filename = os.path.basename(args.out)
        if os.path.splitext(filename)[1] == "":
            filename = filename + ".parquet"
    else:
        base = os.path.splitext(os.path.basename(args.pbf))[0]
        filename = f"{base}_poi.parquet"
    
    out_path = os.path.join(cwd, filename)
    out_df.to_parquet(out_path, index=False)
    print(f"[4] Saved: {out_path}  rows={len(out_df)}")

    print(f"[OK] Done in {tsec(T0)}")

if __name__ == "__main__":
    main()
