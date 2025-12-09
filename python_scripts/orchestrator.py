import argparse, os, sys, csv, subprocess, shlex
from pathlib import Path
from datetime import datetime


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, cwd=None, env=None, timeout=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return proc.returncode, proc.stdout, proc.stderr

def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser(description="Orchestrator: graph_construction -> snap_poi_to_nodes -> precompute_poi_reach for all tiles.")
    ap.add_argument("--tiles-dir", default="tiles", help="Folder with .pbf files (tiles).")
    ap.add_argument("--workspace", default="workspace", help="Output folder for per-tile artifacts.")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to run scripts.")


    ap.add_argument("--graph-script", default="graph_construction.py", help="Path to graph_construction.py")
    ap.add_argument("--snap-script",  default="snap_poi_to_nodes.py", help="Path to snap_poi_to_nodes.py")
    ap.add_argument("--prec-script",  default="precompute_poi_reach.py", help="Path to precompute_poi_reach.py")


    ap.add_argument("--limit-m", type=float, default=1000.0, help="Radius (m) for precompute.")
    ap.add_argument("--speed-mps", type=float, default=1.111, help="Walking speed (m/s) for precompute.")
    ap.add_argument("--cats", nargs="*", default=None, help="List of categories for precompute (optional).")
    ap.add_argument("--skip-plot", action="store_true", help="Pass --no-plot to graph_construction.")
    ap.add_argument("--dump-csv", action="store_true", help="Pass --dump-csv to graph_construction.")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop on first error (default: continue).")
    args = ap.parse_args()

    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.exists():
        print(f"[ERR] tiles dir not found: {tiles_dir}")
        sys.exit(2)

    workspace = Path(args.workspace)
    ensure_dir(workspace)


    summary_path = workspace / "summary.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as fsum:
        w = csv.writer(fsum)
        if write_header:
            w.writerow([
                "tile", "pbf_path",
                "graph_status", "snap_status", "precompute_status",
                "graph_msg", "snap_msg", "precompute_msg",
                "graph_csr", "pois_parquet", "precompute_npz"
            ])

        pbfs = sorted(tiles_dir.glob("*.pbf"))
        if not pbfs:
            print(f"[WARN] No .pbf files in {tiles_dir}")
            sys.exit(0)

        for pbf in pbfs:
            tile_name = pbf.stem  
            tile_dir = workspace / tile_name
            ensure_dir(tile_dir)

            log_file = tile_dir / "run.log"
            with log_file.open("a", encoding="utf-8") as flog:
                def log(msg):
                    line = f"[{stamp()}] {msg}"
                    print(line)
                    print(line, file=flog, flush=True)

                log(f"=== TILE START: {tile_name} ===")
                log(f"PBF: {pbf}")

                out_prefix = str(tile_dir / "graph")
                graph_cmd = [
                    args.python, args.graph_script,
                    "--pbf", str(pbf),
                    "--out-prefix", out_prefix
                ]
                if args.skip_plot:
                    graph_cmd.append("--no-plot")
                if args.dump_csv:
                    graph_cmd.append("--dump-csv")

                log(f"[graph] run: {shlex.join(graph_cmd)}")
                code_g, out_g, err_g = run_cmd(graph_cmd)
                graph_csr = tile_dir / "graph_csr.npz"
                graph_ok = (graph_csr.exists() and graph_csr.stat().st_size > 0)
                graph_msg = f"exit={code_g}"
                if not graph_ok:
                    graph_msg = f"FAILED exit={code_g} | {err_g.strip()[:300]}"
                    log(f"[graph] ERROR: {graph_msg}")
                else:
                    log(f"[graph] OK: {graph_csr}")

                pois_parquet = tile_dir / "pois.parquet"
                if graph_ok:
                    snap_cmd = [
                        args.python, args.snap_script,
                        "--pbf", str(pbf),
                        "--csr", str(graph_csr),
                        "--out", str(pois_parquet)
                    ]
                    log(f"[snap] run: {shlex.join(snap_cmd)}")
                    code_s, out_s, err_s = run_cmd(snap_cmd)

                    snap_ok = (pois_parquet.exists() and pois_parquet.stat().st_size > 0)
                    if snap_ok:
                        if code_s == 0:
                            snap_msg = "OK"
                        else:
                            snap_msg = f"OK (non-zero exit={code_s}) | {err_s.strip().splitlines()[:1]}"
                        log(f"[snap] {snap_msg}: {pois_parquet}")
                    else:
                        snap_msg = f"FAILED exit={code_s} | {err_s.strip()[:300]}"
                        log(f"[snap] ERROR: {snap_msg}")
                else:
                    snap_ok = False
                    snap_msg = "SKIPPED (graph failed)"
                    log("[snap] SKIPPED")

                precompute_npz = tile_dir / "precompute.npz"
                if graph_ok and snap_ok:
                    prec_cmd = [
                        args.python, args.prec_script,
                        "--csr", str(graph_csr),
                        "--pois", str(pois_parquet),
                        "--out-prefix", str(tile_dir / "precompute"),
                        "--limit-m", str(args.limit_m),
                        "--speed-mps", str(args.speed_mps)
                    ]
                    if args.cats:
                        prec_cmd += ["--cats"] + list(args.cats)

                    log(f"[precompute] run: {shlex.join(prec_cmd)}")
                    code_p, out_p, err_p = run_cmd(prec_cmd)

                    pc1 = tile_dir / "precompute_precompute.npz"
                    if pc1.exists():
                        pc1.rename(precompute_npz)

                    prec_ok = (precompute_npz.exists() and precompute_npz.stat().st_size > 0)
                    if prec_ok:
                        prec_msg = f"OK (exit={code_p})"
                        log(f"[precompute] OK: {precompute_npz}")
                    else:
                        prec_msg = f"FAILED exit={code_p} | {err_p.strip()[:300]}"
                        log(f"[precompute] ERROR: {prec_msg}")
                else:
                    prec_ok = False
                    prec_msg = "SKIPPED (graph or snap failed)"
                    log("[precompute] SKIPPED")

                w.writerow([
                    tile_name, str(pbf),
                    "OK" if graph_ok else "ERROR",
                    "OK" if snap_ok else "ERROR",
                    "OK" if prec_ok else "ERROR",
                    graph_msg, snap_msg, prec_msg,
                    str(graph_csr), str(pois_parquet), str(precompute_npz)
                ])
                flog.flush()

                log(f"=== TILE END: {tile_name} ===\n")

                if args.stop_on_error and (not graph_ok or not snap_ok or not prec_ok):
                    print("[orchestrator] stop-on-error enabled -> aborting.")
                    sys.exit(1)

    print(f"[orchestrator] Summary saved to: {summary_path}")
    print("[orchestrator] Done.")

if __name__ == "__main__":
    main()
