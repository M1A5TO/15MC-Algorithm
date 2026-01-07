import argparse, json, sys, subprocess
from pathlib import Path
from datetime import datetime
import logging
import json
import pandas as pd

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log = logging.getLogger("automatic_poi_query")

def point_in_tile_bbox(rec, lat: float, lon: float):
    bbox = rec.get("tile_bbox_wgs84") or rec.get("bbox_wgs84")
    if not bbox:
        return False
    minlon = float(bbox["minlon"]); minlat = float(bbox["minlat"])
    maxlon = float(bbox["maxlon"]); maxlat = float(bbox["maxlat"])
    return (minlon <= lon <= maxlon) and (minlat <= lat <= maxlat)

def pick_tile_early(grid_json: Path, lat: float, lon: float):
    with open(grid_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Grid JSON must be a list of tile records.")
    for rec in data:
        gid = str(rec.get("grid_id", "")).strip()
        if not gid:
            continue
        if point_in_tile_bbox(rec, lat, lon):
            return gid
    return None

def load_pois_map(parquet_path: Path) -> dict[int, tuple[float, float]]:
    df = pd.read_parquet(parquet_path)

    col_id = "poi_id" 
    col_lat = "lat" 
    col_lon = "lon" 

    if not (col_id and col_lat and col_lon):
        raise ValueError(f"pois.parquet missing columns. Have: {list(df.columns)}")

    mp = {}
    for pid, la, lo in zip(df[col_id].astype(int), df[col_lat].astype(float), df[col_lon].astype(float)):
        mp[int(pid)] = (float(la), float(lo))
    return mp


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr

def main():
    ap = argparse.ArgumentParser(
        description="Pick tile by point-in-bbox and run poi_query.py using existing artifacts."
    )
    ap.add_argument("--lat", type=float, required=True, help="Latitude (WGS84)")
    ap.add_argument("--lon", type=float, required=True, help="Longitude (WGS84)")
    ap.add_argument("--grid-json", default="grid_test.json", help="Grid JSON with tile bboxes and grid_id")
    ap.add_argument("--workspace", default="workspace", help="Per-tile artifacts root (expects workspace/<grid_id>/...)")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to run poi_query.py")
    ap.add_argument("--poi-query-script", default="poi_query.py", help="Path to poi_query.py")
    ap.add_argument("--radius-m", type=float, default=1000.0, help="Query radius in meters (forwarded to poi_query)")
    ap.add_argument("--cats", nargs="*", default=None, help="Optional categories (forwarded to poi_query)")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    grid_id = pick_tile_early(Path(args.grid_json), args.lat, args.lon)
    if not grid_id:
        log.warning(
            "No tile bbox contains point (%.6f, %.6f). Nothing to do.",
            args.lat, args.lon
        )
        sys.exit(4)

    log.info("Selected tile: %s (point is inside tile bbox)", grid_id)

    tile_dir = Path(args.workspace) / grid_id
    csr = tile_dir / "graph_csr.npz"
    precompute = tile_dir / "precompute.npz"
    pois = tile_dir / "pois.parquet" 

    missing = [p for p in [csr, precompute, pois] if not p.exists() or p.stat().st_size == 0]
    if missing:
        log.error(
            "Missing required artifacts for %s: %s",
            grid_id, ", ".join(str(m) for m in missing)
        )
        log.info("Hint: Build them first with your orchestrator, then re-run this query.")
        sys.exit(5)

    cmd = [
        args.python, args.poi_query_script,
        "--csr", str(csr),
        "--precompute", str(precompute),
        "--lat", str(args.lat),
        "--lon", str(args.lon),
        "--json"
    ]
    if args.cats:
        cmd += ["--cats"] + list(args.cats)

    log.info("Running poi_query: %s", " ".join(cmd))
    code, out, err = run(cmd)


    if out:
        payload = json.loads(out)  
        pois_map = load_pois_map(pois)

        items = payload.get("pois_in_range") or payload.get("pois") or []

        for it in items:
            pid = it.get("poi_id")
            if pid is None:
                continue
            pid = int(pid)
            geo = pois_map.get(pid)
            if geo is not None:
                it["geolocation"] = geo  

        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.flush()

    if code != 0:
        msg_lines = (err or "").strip().splitlines()[:5]
        log.error("poi_query FAILED (exit=%s) | %s", code, " | ".join(msg_lines))
        sys.exit(code)

    log.info("poi_query OK (exit=0)")

if __name__ == "__main__":
    main()
