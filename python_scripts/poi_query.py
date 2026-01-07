"""
Query for POI availability from a given location (WGS84 lat/lon) based on:
- CSR .npz (with the “lonlat” key [N,2] in the order [lon, lat])
- precompute .npz (with the keys: cat_keys, cat_names and dist_*, time_*, poi_*)

If --json is provided, prints ONLY JSON to stdout.
All human-readable output goes to logging (stderr).
"""

import argparse
import json
import logging
import math
from typing import Tuple, List, Dict, Any, Optional

import numpy as np


log = logging.getLogger("poi_query")


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
    dist2 = x * x + y * y
    i = int(np.argmin(dist2))

    lat1 = np.deg2rad(lat0)
    lon1 = np.deg2rad(lon0)
    lat2 = np.deg2rad(lat_nodes[i])
    lon2 = np.deg2rad(lon_nodes[i])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    h = 2 * R * math.asin(min(1.0, math.sqrt(a)))
    return i, float(h)


def load_precompute(pre_path: str):
    npz = np.load(pre_path, allow_pickle=True)
    if "cat_keys" in npz and "cat_names" in npz:
        cat_keys = list(npz["cat_keys"].tolist())
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
    ap.add_argument("--json", action="store_true", help="Output ONLY JSON to stdout (logs go to stderr)")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    csr = np.load(args.csr)
    if "lonlat" not in csr:
        raise ValueError("CSR file does not contain 'lonlat'. Without it we cannot map lat/lon -> node.")
    lonlat = csr["lonlat"]

    node_idx, snap_m = nearest_node_idx(lonlat, args.lat, args.lon)

    pre, cat_keys, cat_names = load_precompute(args.precompute)
    if args.cats:
        wanted = set(args.cats)
        mask = [nm in wanted for nm in cat_names]
        cat_keys = [ck for ck, m in zip(cat_keys, mask) if m]
        cat_names = [cn for cn, m in zip(cat_names, mask) if m]

    log.info("Nearest node: %s | snap ~ %.1f m", node_idx, snap_m)
    log.info("Number of categories: %s", len(cat_keys))

    rows_in: List[Tuple[str, float, float, int]] = []
    rows_out: List[str] = []

    for nm, ck in zip(cat_names, cat_keys):
        dist_key = f"dist_{ck}"
        time_key = f"time_{ck}"
        poi_key = f"poi_{ck}"
        if dist_key not in pre or time_key not in pre or poi_key not in pre:
            rows_out.append(nm + " (no data)")
            continue

        d = float(pre[dist_key][node_idx])
        t = float(pre[time_key][node_idx])
        pid = int(pre[poi_key][node_idx])

        if math.isfinite(d) and d <= args.radius_m:
            rows_in.append((nm, d, t, pid))
        else:
            rows_out.append(nm)

    rows_in.sort(key=lambda r: r[1])

    if args.json:
        result: Dict[str, Any] = {
            "input": {
                "lat": float(args.lat),
                "lon": float(args.lon)
            },
            "pois_in_range": [
                {
                    "category": nm,
                    "time_to_poi": int(round(t)),
                    "poi_id": int(pid),
                    "geolocation": {"lat": None, "lng": None},  
                }
                for (nm, d, t, pid) in rows_in
            ],
            "pois_out_of_range": rows_out,
        }
        print(json.dumps(result, ensure_ascii=False))
        return

    if rows_in:
        log.info("=== POIs within %.0f m ===", args.radius_m)
        log.info("%-20s %10s %10s %14s", "category", "distance[m]", "time[s]", "poi_id")
        log.info("%s", "-" * 60)
        for nm, d, t, pid in rows_in:
            log.info("%-20s %10.1f %10.0f %14d", nm, d, t, pid)
    else:
        log.info("No POIs within the specified radius for this location.")

    if rows_out:
        log.info("(Not within range): %s", ", ".join(rows_out))


if __name__ == "__main__":
    main()
