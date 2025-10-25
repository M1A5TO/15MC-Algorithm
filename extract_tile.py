"""
extract_tile.py

Usage example:
  python extract_tile.py --pbf /path/to/poland.pbf --lat 54.3520 --lon 18.6466 \
    --radius 10000 --out gdynia_10km.pbf

Script:
 - calculates a bbox with the given radius around lat/lon
 - runs `osmium extract -b min_lon,min_lat,max_lon,max_lat -s complete_ways ...`
 - if `osmium` is not found, suggests a Docker command to run (optional fallback)
"""
import argparse
import math
import shutil
import subprocess
import os
import sys

def bbox_from_point(lat, lon, radius_m):
    # returns (min_lon, min_lat, max_lon, max_lat)
    R = 6378137.0
    lat_rad = math.radians(lat)
    dlat = (radius_m / R) * (180.0 / math.pi)
    dlon = (radius_m / (R * math.cos(lat_rad))) * (180.0 / math.pi)
    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon
    return (min_lon, min_lat, max_lon, max_lat)

def check_osmium():
    return shutil.which("osmium") is not None

def run_osmium_extract(osmium_path, src_pbf, dest_pbf, bbox, extra_opts=None):
    min_lon, min_lat, max_lon, max_lat = bbox
    # build command
    cmd = [
        osmium_path, "extract",
        "-b", f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "-s", "complete_ways",
        "--set-bounds",
        "-o", dest_pbf,
        src_pbf
    ]
    if extra_opts:
        # insert extra_opts before '-o' and src_pbf
        cmd = cmd[:-2] + extra_opts + cmd[-2:]
    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("osmium extract returned an error:", e)
        raise

def docker_osmium_command(src_pbf, dest_pbf, bbox, workdir):
    min_lon, min_lat, max_lon, max_lat = bbox
    # mount working directory to /data in the container
    # assumes src_pbf is accessible under workdir. We'll mount absolute workdir.
    cmd = (
        "docker run --rm -v {wd}:/data osmcode/osmium-tool osmium extract "
        "-b {min_lon},{min_lat},{max_lon},{max_lat} -s complete_ways --set-bounds "
        "-o /data/{dest} /data/{src}"
    ).format(
        wd=workdir,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        dest=os.path.basename(dest_pbf),
        src=os.path.basename(src_pbf)
    )
    return cmd

def main():
    p = argparse.ArgumentParser(description="Extract a bbox (radius) from a master .pbf using osmium.")
    p.add_argument("--pbf", required=True, help="Path to the master .pbf (e.g. poland.pbf)")
    p.add_argument("--lat", type=float, required=True, help="Latitude of the center point")
    p.add_argument("--lon", type=float, required=True, help="Longitude of the center point")
    p.add_argument("--radius", type=float, default=10000, help="Radius in meters (default 10000 = 10km)")
    p.add_argument("--out", required=True, help="Path to the resulting pbf (e.g. tile_10km.pbf)")
    p.add_argument("--no-docker-fallback", action="store_true", help="Do not suggest a Docker fallback if osmium is missing")
    p.add_argument("--relations", action="store_true", help="Include relations in the extract (may increase size)")
    args = p.parse_args()

    if not os.path.isfile(args.pbf):
        print("Source .pbf file not found:", args.pbf)
        sys.exit(2)

    bbox = bbox_from_point(args.lat, args.lon, args.radius)
    print("Computed bbox (min_lon,min_lat,max_lon,max_lat):", bbox)
    print(f"Source PBF: {args.pbf}")
    print(f"Output file: {args.out}")

    if check_osmium():
        osmium_bin = shutil.which("osmium")
        extra = []
        if not args.relations:
            # turn off relations if not requested (saves size/memory)
            extra = ["-S", "relations=false"]
        try:
            run_osmium_extract(osmium_bin, args.pbf, args.out, bbox, extra_opts=extra)
            print("Done! Saved:", args.out)
            try:
                print("Output file size:", os.path.getsize(args.out), "bytes")
            except Exception:
                pass
            sys.exit(0)
        except Exception:
            print("Error while running osmium locally.")
    else:
        print("`osmium` not found in PATH.")

    # fallback: docker suggestion
    if args.no_docker_fallback:
        print("No local osmium found and Docker fallback is disabled.")
        sys.exit(3)

    wd = os.path.abspath(os.path.dirname(args.pbf))
    docker_cmd = docker_osmium_command(args.pbf, args.out, bbox, wd)
    print("\nYou can run the extract using Docker (if you have Docker installed):\n")
    print(docker_cmd)
    print("\nNote: Docker will mount the working directory; make sure the source file is accessible there.")
    print("If you want, run the above Docker command in your shell.")
    sys.exit(4)

if __name__ == "__main__":
    main()
