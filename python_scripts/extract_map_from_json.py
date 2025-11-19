import argparse, json, os, sys, shutil, subprocess
from typing import List, Dict, Tuple, Optional


def check_osmium() -> Optional[str]:
    return shutil.which("osmium")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON should be a list of cells/tiles.")
    return data

def parse_grid_file(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = []
            for tok in line.replace(",", " ").split():
                tok = tok.strip()
                if tok:
                    parts.append(tok)
            ids.extend(parts)

    seen = set()
    unique_ids = []
    for gid in ids:
        if gid not in seen:
            seen.add(gid)
            unique_ids.append(gid)
    return unique_ids


def pick_tiles(cells: List[Dict], grid_ids: List[str], limit: int) -> List[Dict]:
    if grid_ids:
        wanted = set(grid_ids)
        sel = [c for c in cells if str(c.get("grid_id")) in wanted]
    else:
        sel = cells[:limit] if limit > 0 else cells
    if not sel:
        raise ValueError("No tiles selected (check --grid-ids or --limit).")
    return sel

def bbox_from_record(rec: Dict, mode: str) -> Tuple[float, float, float, float]:
    key = "buffer_bbox_wgs84" if mode == "buffer" else "tile_bbox_wgs84"
    if key not in rec:
        key = "bbox_wgs84"
        if key not in rec:
            raise KeyError(f"Record missing '{key}' bbox.")
    b = rec[key]
    return (b["minlon"], b["minlat"], b["maxlon"], b["maxlat"])

def run_osmium_extract(osmium_path: str, src_pbf: str, dest_pbf: str,
                       bbox: Tuple[float, float, float, float],
                       include_relations: bool) -> int:
    minlon, minlat, maxlon, maxlat = bbox
    cmd = [
        osmium_path, "extract",
        "-b", f"{minlon},{minlat},{maxlon},{maxlat}",
        "-s", "complete_ways",
        "--set-bounds",
    ]
    if not include_relations:
        cmd += ["-S", "relations=false"]
    cmd += ["-o", dest_pbf, src_pbf]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)

def docker_osmium_cmd(src_pbf: str, dest_rel: str, bbox: Tuple[float, float, float, float],
                      include_relations: bool, workdir: str) -> str:
    minlon, minlat, maxlon, maxlat = bbox
    rel_flag = "" if include_relations else "-S relations=false"
    return (
        f"docker run --rm -v {workdir}:/data osmcode/osmium-tool "
        f"osmium extract -b {minlon},{minlat},{maxlon},{maxlat} -s complete_ways --set-bounds "
        f"{rel_flag} -o /data/{dest_rel} /data/{os.path.basename(src_pbf)}"
    )

def osmium_count(osmium_path: str, pbf_path: str) -> Tuple[int, int, int]:
    try:
        out = subprocess.check_output([osmium_path, "count", pbf_path], stderr=subprocess.STDOUT, text=True)
        n = w = r = 0
        for line in out.splitlines():
            line = line.strip().lower()
            if line.startswith("nodes:"):
                n = int(line.split(":")[1].strip().split()[0])
            elif line.startswith("ways:"):
                w = int(line.split(":")[1].strip().split()[0])
            elif line.startswith("relations:"):
                r = int(line.split(":")[1].strip().split()[0])
        return (n, w, r)
    except Exception:
        return (-1, -1, -1)


def main():
    ap = argparse.ArgumentParser
    ap.add_argument("--pbf", required=True, help="Path to source .pbf (e.g., poland-latest.osm.pbf)")
    ap.add_argument("--json", required=True, help="Grid JSON produced earlier (with tile_bbox_wgs84 & buffer_bbox_wgs84)")
    ap.add_argument("--out-dir", required=True, help="Output directory for extracted .pbf files")
    ap.add_argument("--use", choices=["tile", "buffer"], default="buffer", help="Which bbox to use for extraction (default: buffer)")
    ap.add_argument("--limit", type=int, default=4, help="How many tiles to process if --grid-ids not given (default: 4)")
    ap.add_argument("--grid-file", default="", help="Path to a text file with grid IDs (one per line or comma/space-separated)")
    ap.add_argument("--relations", action="store_true", help="Include relations in extracts (bigger/fewer empties).")
    ap.add_argument("--delete-empty", action="store_true", help="Delete extracts with 0 nodes/ways/relations.")
    ap.add_argument("--no-docker-fallback", action="store_true", help="Do not print Docker commands if osmium is missing.")
    args = ap.parse_args()

    if not os.path.isfile(args.pbf):
        print(f"Source .pbf not found: {args.pbf}")
        sys.exit(2)

    try:
        cells = parse_json(args.json)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        sys.exit(2)

    grid_ids: List[str] = []
    if args.grid_file:
        if not os.path.isfile(args.grid_file):
            print(f"Grid file not found: {args.grid_file}")
            sys.exit(2)
        try:
            grid_ids = parse_grid_file(args.grid_file)
            if not grid_ids:
                print(f"No grid IDs parsed from file: {args.grid_file}")
                sys.exit(2)
            print(f"Loaded {len(grid_ids)} grid IDs from {args.grid_file}")
        except Exception as e:
            print(f"Failed to parse grid file: {e}")
            sys.exit(2)
    
    tiles = pick_tiles(cells, grid_ids, args.limit)
    ensure_dir(args.out_dir)

    osmium_path = check_osmium()
    docker_cmds: List[str] = []

    for rec in tiles:
        grid_id = str(rec.get("grid_id", "tile"))
        bbox = bbox_from_record(rec, args.use)
        outfile = os.path.join(args.out_dir, f"{grid_id}.pbf")

        if osmium_path:
            rc = run_osmium_extract(osmium_path, args.pbf, outfile, bbox, include_relations=args.relations)
            if rc != 0:
                print(f"[{grid_id}] osmium extract failed with code {rc}")
                continue

            if args.delete_empty:
                n, w, r = osmium_count(osmium_path, outfile)
                if (n, w, r) == (0, 0, 0):
                    try:
                        os.remove(outfile)
                        print(f"[{grid_id}] Empty extract (0 nodes/ways/relations) — removed.")
                    except Exception:
                        pass
                else:
                    print(f"[{grid_id}] Counts: nodes={n}, ways={w}, relations={r}")
        else:
            if args.no_docker_fallback:
                print("`osmium` not found and Docker fallback disabled. Aborting.")
                sys.exit(3)
            workdir = os.path.abspath(args.out_dir)
            src_in_mount = os.path.abspath(args.pbf)
            if os.path.dirname(src_in_mount) != workdir:
                print(f"Note: For Docker one-liner, place source PBF inside {workdir} or adjust path.")
            dest_rel = os.path.basename(outfile)
            docker_cmds.append(docker_osmium_cmd(args.pbf, dest_rel, bbox, args.relations, workdir))

    if (not osmium_path) and docker_cmds:
        print("\n`osmium` not found. You can run the following Docker commands to perform extracts:")
        for c in docker_cmds:
            print("  " + c)

if __name__ == "__main__":
    main()
