import json
import sys
from collections import OrderedDict

def main(json_path="grid_test.json", out_path="grid_ids.txt"):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("grid JSON should be object list (cells/tiles).")

    ids = []
    seen = set()
    for rec in data:
        gid = str(rec.get("grid_id", "")).strip()
        if not gid:
            continue
        if gid not in seen:
            seen.add(gid)
            ids.append(gid)

    if not ids:
        print("grid_id not found in JSON.", file=sys.stderr)
        return

    with open(out_path, "w", encoding="utf-8") as f:
        for gid in ids:
            f.write(gid + "\n")

    print(f"Saved {len(ids)} grid_id to: {out_path}")

if __name__ == "__main__":
    json_in = sys.argv[1] if len(sys.argv) > 1 else "grid_test.json"
    out_txt = sys.argv[2] if len(sys.argv) > 2 else "grid_ids.txt"
    main(json_in, out_txt)