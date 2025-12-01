import argparse, json, sys, subprocess
from pathlib import Path
from datetime import datetime

def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr

def main():
    ap = argparse.ArgumentParser(description="Pick tile by point-in-bbox and run poi_query.py using existing artifacts.")
    ap.add_argument("--lat", type=float, required=True, help="Latitude (WGS84)")
    ap.add_argument("--lon", type=float, required=True, help="Longitude (WGS84)")
    ap.add_argument("--grid-json", default="grid_test.json", help="Grid JSON with tile bboxes and grid_id")
    ap.add_argument("--workspace", default="workspace", help="Per-tile artifacts root (expects workspace/<grid_id>/...)")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to run poi_query.py")
    ap.add_argument("--poi-query-script", default="poi_query.py", help="Path to poi_query.py")
    ap.add_argument("--radius-m", type=float, default=1000.0, help="Query radius in meters (forwarded to poi_query)")
    ap.add_argument("--cats", nargs="*", default=None, help="Optional categories (forwarded to poi_query)")
    args = ap.parse_args()

    grid_id = pick_tile_early(Path(args.grid_json), args.lat, args.lon)
    if not grid_id:
        print(f"[{ts()}] No tile bbox contains point ({args.lat:.6f}, {args.lon:.6f}). Nothing to do.")
        sys.exit(4)

    print(f"[{ts()}] Selected tile: {grid_id} (point is inside tile bbox)")

    tile_dir = Path(args.workspace) / grid_id
    csr = tile_dir / "graph_csr.npz"
    precompute = tile_dir / "precompute.npz"
    pois = tile_dir / "pois.parquet" 

    missing = [p for p in [csr, precompute] if not p.exists() or p.stat().st_size == 0]
    if missing:
        print(f"[{ts()}] Missing required artifacts for {grid_id}: "
              + ", ".join(str(m) for m in missing))
        print(f"[hint] Build them first with your orchestrator, then re-run this query.")
        sys.exit(5)

    cmd = [
        args.python, args.poi_query_script,
        "--csr", str(csr),
        "--precompute", str(precompute),
        "--lat", str(args.lat),
        "--lon", str(args.lon),
        "--radius-m", str(args.radius_m),
    ]
    if args.cats:
        cmd += ["--cats"] + list(args.cats)

    print(f"[{ts()}] Running poi_query: {' '.join(cmd)}")
    code, out, err = run(cmd)


    if out:
        print(out, end="")  
    if code != 0:
        msg = err.strip().splitlines()[:5]
        print(f"[{ts()}] poi_query FAILED (exit={code}) | " + " | ".join(msg), file=sys.stderr)
        sys.exit(code)

    print(f"[{ts()}] poi_query OK (exit=0)")

if __name__ == "__main__":
    main()
