"""
Precompute multi-source Dijkstra per category (limit zasięgu) na grafie CSR.

Wyjście:
  {out_prefix}_precompute.npz  zawiera dla każdej kategorii (po kluczu zsanityzowanym):
    - dist_{cat_key}: float32 [N]  dystans w metrach (inf gdy poza limitem)
    - time_{cat_key}: float32 [N]  czas w sekundach (inf gdy poza limitem)
    - poi_{cat_key}:  int64   [N]  OSM poi_id najbliższego POI (-1 gdy brak)
"""

import argparse
import math
import os
import re
import time
import heapq
import numpy as np
import pandas as pd

def tsec(t0): return f"{(time.perf_counter()-t0):.3f}s"

def sanitize_key(name: str) -> str:

    if name is None:
        return "cat"
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "cat"
    return s[:60] 

def load_pois(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".parquet":
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                raise RuntimeError(
                    "Nie udało się wczytać .parquet. Zainstaluj 'pyarrow' lub 'fastparquet', "
                    "albo zapisz POI jako CSV (z nagłówkami: poi_id, category, node_idx)."
                ) from e
        else:
            df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Brak pliku POI: {path}")

    cols = {c.lower(): c for c in df.columns}
    need = {"poi_id","category","node_idx"}
    if not need.issubset(set(c.lower() for c in df.columns)):
        raise ValueError("POI file must have columns: poi_id, category, node_idx")

    df = df.rename(columns={
        cols["poi_id"]: "poi_id",
        cols["category"]: "category",
        cols["node_idx"]: "node_idx",
    })
    df["poi_id"] = pd.to_numeric(df["poi_id"], errors="coerce").fillna(-1).astype(np.int64)
    df["category"] = df["category"].astype(str)
    df["node_idx"] = pd.to_numeric(df["node_idx"], errors="coerce").fillna(-1).astype(np.int64)
    return df

def load_csr(path):
    try:
        csr = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Brak pliku CSR: {path}")
    keys = set(csr.keys())
    for k in ("indptr","indices"):
        if k not in keys:
            raise ValueError(f"CSR .npz musi zawierać '{k}'")

    indptr  = csr["indptr"].astype(np.int32, copy=False)
    indices = csr["indices"].astype(np.int32, copy=False)
    if "weights" in keys:
        weights = csr["weights"].astype(np.float32, copy=False)
    elif "data" in keys:
        weights = csr["data"].astype(np.float32, copy=False)
    else:
        raise ValueError("CSR .npz musi zawierać 'weights' (lub alternatywnie 'data').")
    N = indptr.shape[0] - 1
    if len(indices) != len(weights):
        raise ValueError("CSR niespójny: len(indices) != len(weights)")
    return indptr, indices, weights, N

def dijkstra_multi_source_idx(indptr, indices, weights, sources_idx, limit_m):

    N = indptr.shape[0] - 1
    INF = np.float32(np.inf)

    dist = np.full(N, INF, dtype=np.float32)
    srci = np.full(N, -1, dtype=np.int32)
    hq = []

    for nidx, sidx in sources_idx:
        nidx_i = int(nidx)
        sidx_i = int(sidx)
        if 0 <= nidx_i < N and sidx_i >= 0:
            if dist[nidx_i] > 0:
                dist[nidx_i] = 0.0
                srci[nidx_i] = sidx_i
                heapq.heappush(hq, (0.0, nidx_i, sidx_i))

    L = np.float32(limit_m)

    while hq:
        d_u, u, s_idx = heapq.heappop(hq)
        if d_u > dist[u]:
            continue
        if d_u > L:
            break  
        start, end = indptr[u], indptr[u+1]
        for i in range(start, end):
            v = int(indices[i])
            nd = d_u + weights[i]
            if nd < dist[v] and nd <= L:
                dist[v] = nd
                srci[v] = s_idx
                heapq.heappush(hq, (float(nd), v, s_idx))

    return dist, srci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", required=True, help="CSR .npz z grafem (indptr, indices, weights|data, ...)")
    ap.add_argument("--pois", required=True, help="POI parquet/csv: poi_id, category, node_idx")
    ap.add_argument("--out-prefix", required=True, help="Prefix plików wyjściowych (np. data/gdansk)")
    ap.add_argument("--limit-m", type=float, default=None, help="Limit promienia w metrach (np. 1000)")
    ap.add_argument("--limit-min", type=float, default=None, help="Limit czasu w minutach (np. 15)")
    ap.add_argument("--speed-mps", type=float, default=1.4, help="Prędkość piesza [m/s] (domyślnie 1.4)")
    ap.add_argument("--cats", nargs="*", default=None, help="Konkretne kategorie (domyślnie wszystkie z POI)")
    ap.add_argument("--no-summary", action="store_true", help="Nie generuj .csv z podsumowaniem")
    args = ap.parse_args()

    T0 = time.perf_counter()
    print(f"[0] Start | csr={args.csr}  pois={args.pois}")

    t = time.perf_counter()
    indptr, indices, weights, N = load_csr(args.csr)
    print(f"[1] CSR: N={N}, E={len(indices)}  ({tsec(t)})")

    if args.limit_m is None and args.limit_min is None:
        limit_m = 1000.0
        print("[i] Brak limitu -> domyślnie 1000 m")
    elif args.limit_m is not None:
        limit_m = float(args.limit_m)
    else:
        limit_m = float(args.limit_min) * 60.0 * float(args.speed_mps)
    print(f"[i] LIMIT: {limit_m:.1f} m (speed={args.speed_mps} m/s)")

    t = time.perf_counter()
    poi_df = load_pois(args.pois)
    if args.cats:
        poi_df = poi_df[poi_df["category"].isin(args.cats)].copy()
    cats_names = sorted(poi_df["category"].unique().tolist())
    cats_keys  = [sanitize_key(c) for c in cats_names]
    print(f"[2] POI: {len(poi_df)}  kategorie={cats_names}  ({tsec(t)})")

    OUT = {}
    summary = []

    for cat_name, cat_key in zip(cats_names, cats_keys):
        tcat = time.perf_counter()
        sub = poi_df[poi_df["category"] == cat_name]
        if sub.empty:
            print(f"[{cat_name}] brak źródeł – pomijam.")
            dist = np.full(N, np.float32(np.inf), dtype=np.float32)
            src_idx = np.full(N, -1, dtype=np.int32)
            time_s = dist / float(args.speed_mps)
            poi_ids = np.full(N, -1, dtype=np.int64)
            OUT[f"dist_{cat_key}"] = dist
            OUT[f"time_{cat_key}"] = time_s
            OUT[f"poi_{cat_key}"]  = poi_ids
            summary.append((cat_name, 0, 0, math.inf, math.inf, limit_m))
            continue

        cat_poi_ids = sub["poi_id"].to_numpy(dtype=np.int64, copy=True)
        cat_nodes   = sub["node_idx"].to_numpy(dtype=np.int64, copy=True)

        valid_mask = (cat_nodes >= 0) & (cat_nodes < N)
        cat_nodes = cat_nodes[valid_mask]
        cat_poi_ids = cat_poi_ids[valid_mask]
        if len(cat_nodes) == 0:
            print(f"[{cat_name}] brak poprawnych źródeł – pomijam.")
            dist = np.full(N, np.float32(np.inf), dtype=np.float32)
            src_idx = np.full(N, -1, dtype=np.int32)
            time_s = dist / float(args.speed_mps)
            poi_ids = np.full(N, -1, dtype=np.int64)
            OUT[f"dist_{cat_key}"] = dist
            OUT[f"time_{cat_key}"] = time_s
            OUT[f"poi_{cat_key}"]  = poi_ids
            summary.append((cat_name, 0, 0, math.inf, math.inf, limit_m))
            continue

        K = len(cat_nodes)
        sources_idx = list(zip(cat_nodes.astype(np.int32), np.arange(K, dtype=np.int32)))

        print(f"[{cat_name}] źródeł: {K} — Dijkstra...")
        dist, src_idx = dijkstra_multi_source_idx(indptr, indices, weights, sources_idx, limit_m)
        time_s = dist / float(args.speed_mps)

        poi_ids = np.full(N, -1, dtype=np.int64)
        mask = src_idx >= 0
        if mask.any():
            poi_ids[mask] = cat_poi_ids[src_idx[mask]]

        within = np.isfinite(dist)
        cnt_within = int(within.sum())
        med_d = float(np.median(dist[within])) if cnt_within>0 else math.inf
        med_t = float(np.median(time_s[within])) if cnt_within>0 else math.inf
        summary.append((cat_name, K, cnt_within, med_d, med_t, limit_m))

        OUT[f"dist_{cat_key}"] = dist.astype(np.float32, copy=False)
        OUT[f"time_{cat_key}"] = time_s.astype(np.float32, copy=False)
        OUT[f"poi_{cat_key}"]  = poi_ids  # int64

        print(f"[{cat_name}] done in {tsec(tcat)} | within={cnt_within}/{N} | med_d={med_d:.1f} m | med_t={med_t/60:.1f} min")

    t = time.perf_counter()
    OUT["cat_keys"]  = np.array(cats_keys,  dtype=object)
    OUT["cat_names"] = np.array(cats_names, dtype=object)
    np.savez_compressed(f"{args.out_prefix}_precompute.npz", **OUT)
    print(f"[3] Zapis: {args.out_prefix}_precompute.npz  ({tsec(t)})")

    if not args.no_summary:
        sum_df = pd.DataFrame(
            summary,
            columns=["category","n_sources","n_within","median_dist_m","median_time_s","limit_m"]
        )
        sum_df.to_csv(f"{args.out_prefix}_precompute_summary.csv", index=False)
        print(f"[4] Podsumowanie: {args.out_prefix}_precompute_summary.csv")

    print(f"[OK] Całość: {tsec(T0)}")

if __name__ == "__main__":
    main()
