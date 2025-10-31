"""
Query for POI availability from a given location (WGS84 lat/lon) based on:
- CSR .npz (with the “lonlat” key [N,2] in the order [lon, lat])
- precompute .npz (with the keys: cat_keys, cat_names and dist_*, time_*, poi_*)

Example:
python scripts/query_poi_at_location.py ^
--csr data\gdansk_full_data_grid_10x10_MD_csr.npz ^
--precompute data\gdansk_precompute.npz ^
    --lat 54.34805179757348 --lon 18.65299820209096 ^
    --radius-m 1000

Result: table with categories within range and basic metrics.
"""

import argparse
import numpy as np
import math
from typing import Tuple, Dict, List

def nearest_node_idx(lonlat: np.ndarray, lat: float, lon: float) -> Tuple[int, float]:
 
    lon_nodes = lonlat[:, 0].astype(np.float64)
    lat_nodes = lonlat[:, 1].astype(np.float64)

    lat0 = float(lat)
    lon0 = float(lon)

    R = 6371000.0  
    lat0r = np.deg2rad(lat0)
    dlon = np.deg2rad(lon_nodes - lon0)
    dlat = np.deg2rad(lat_nodes - lat0)
    x = dlon * np.cos(lat0r)
    y = dlat
    dist2 = x*x + y*y
    i = int(np.argmin(dist2))

    lat1 = np.deg2rad(lat0)
    lon1 = np.deg2rad(lon0)
    lat2 = np.deg2rad(lat_nodes[i])
    lon2 = np.deg2rad(lon_nodes[i])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
    h = 2*R*math.asin(min(1.0, math.sqrt(a)))
    return i, float(h)

def load_precompute(pre_path: str):
    npz = np.load(pre_path, allow_pickle=True)
    if "cat_keys" in npz and "cat_names" in npz:
        cat_keys  = list(npz["cat_keys"].tolist())
        cat_names = list(npz["cat_names"].tolist())
    else:
        all_keys = [k for k in npz.keys() if k.startswith("dist_")]
        cat_keys = [k[len("dist_"):] for k in sorted(all_keys)]
        cat_names = cat_keys[:]
    return npz, cat_keys, cat_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", required=True, help="Path to CSR .npz (must contain 'lonlat')")
    ap.add_argument("--precompute", required=True, help="Path to precompute .npz")
    ap.add_argument("--lat", type=float, required=True, help="Latitude (WGS84)")
    ap.add_argument("--lon", type=float, required=True, help="Longitude (WGS84)")
    ap.add_argument("--radius-m", type=float, default=1000.0, help="Query radius in meters (default 1000)")
    ap.add_argument("--cats", nargs="*", default=None, help="Filter by categories (names from cat_names)")
    args = ap.parse_args()

    csr = np.load(args.csr)
    if "lonlat" not in csr:
        raise ValueError("CSR file does not contain 'lonlat'. Without it we cannot map lat/lon -> node.")
    lonlat = csr["lonlat"]

    node_idx, snap_m = nearest_node_idx(lonlat, args.lat, args.lon)

    pre, cat_keys, cat_names = load_precompute(args.precompute)
    if args.cats:
        mask = [nm in set(args.cats) for nm in cat_names]
        cat_keys  = [ck for ck, m in zip(cat_keys, mask) if m]
        cat_names = [cn for cn, m in zip(cat_names, mask) if m]

    print(f"[i] Nearest node: {node_idx} | snap ~ {snap_m:.1f} m")
    print(f"[i] Number of categories: {len(cat_keys)}")

    rows_in: List[Tuple[str, float, float, int]] = []
    rows_out: List[str] = []

    for nm, ck in zip(cat_names, cat_keys):
        dist_key = f"dist_{ck}"
        time_key = f"time_{ck}"
        poi_key  = f"poi_{ck}"
        if dist_key not in pre or time_key not in pre or poi_key not in pre:
            rows_out.append(nm + " (no data)")
            continue

        d = float(pre[dist_key][node_idx])     
        t = float(pre[time_key][node_idx])    
        pid = int(pre[poi_key][node_idx])     

        if math.isfinite(d) and d <= args.radius_m:
            rows_in.append((nm, d, t/60.0, pid))
        else:
            rows_out.append(nm)

    rows_in.sort(key=lambda r: r[1])

    if rows_in:
        print("\n=== POIs within {:.0f} m ===".format(args.radius_m))
        print("{:<20} {:>10} {:>10} {:>14}".format("category", "distance[m]", "time[min]", "poi_id"))
        print("-"*60)
        for nm, d, tmin, pid in rows_in:
            print("{:<20} {:>10.1f} {:>10.2f} {:>14}".format(nm, d, tmin, pid))
    else:
        print("\nNo POIs within the specified radius for this location.")

    if rows_out:
        print("\n(Not within range): " + ", ".join(rows_out))

if __name__ == "__main__":
    main()
