
"""
Użycie:
  python graph_construction.py --pbf data/gdansk_5km.pbf --out-prefix walk_clean
  python scripts/graph_construction.py --pbf data/gdynia_5km.pbf --out-prefix test --no-plot
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyrosm import OSM
from scipy.spatial import cKDTree
from shapely import wkt

def build_adjacency(edges_df, walk_speed_m_s=1.4):
    adj = defaultdict(list)
    for idx, row in edges_df.iterrows():
        try:
            u = int(row["u"]); v = int(row["v"])
        except Exception:
            continue
        length = float(row.get("length_m", 0.0)) if row.get("length_m") is not None else 0.0
        time_s = length / walk_speed_m_s if length and walk_speed_m_s else None
        edge_id = int(row.get("way_id", idx))
        attr = {
            "edge_id": edge_id,
            "length_m": length,
            "cost_s": time_s,
            "highway": row.get("highway"),
        }
        adj[u].append({"to": v, **attr})
        oneway = str(row.get("oneway")).lower()
        if oneway not in ("yes", "true", "1"):
            adj[v].append({"to": u, **attr})
    return adj


def _safe_wkt_coords(geom):
    if geom is None:
        return None
    try:
        return list(geom.coords)
    except Exception:
        try:
            g = wkt.loads(geom)
            return list(g.coords)
        except Exception:
            return None


def _norm(s):
    return str(s).strip().lower() if s is not None else ""


def clean_walkable_edges(edges_gdf):
    """Zastosuj reguły 'walkable' opisane w rozmowie."""
    edges = edges_gdf.copy()

    for col in ("highway", "foot", "sidewalk", "motorroad", "oneway"):
        if col in edges.columns:
            edges[col + "_n"] = edges[col].map(_norm)
        else:
            edges[col + "_n"] = ""

    always_walkable = {
        "footway", "path", "pedestrian", "steps", "platform", "crossing",
        "living_street", "cycleway", "track", "residential", "unclassified", "service",
    }

    big_roads = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}

    has_sidewalk = edges["sidewalk_n"].isin({"yes", "both", "left", "right"})
    has_foot_ok = edges["foot_n"].isin({"yes", "designated", "permissive"})

    is_fast = edges["highway_n"].isin({"motorway", "motorway_link", "trunk", "trunk_link"}) | (edges["motorroad_n"] == "yes")

    walkable_mask = (
        edges["highway_n"].isin(always_walkable)
        | (edges["highway_n"].isin(big_roads) & (has_sidewalk | has_foot_ok))
    )

    cleaned = edges[walkable_mask & (~is_fast)].copy()
    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True, help="Ścieżka do .osm.pbf (najlepiej już przefiltrowanego osmium).")
    ap.add_argument("--out-prefix", default="walk_clean", help="prefix plików wyjściowych CSV/JSON/PNG")
    ap.add_argument("--no-plot", action="store_true", help="Pomiń rysowanie (szybciej na serwerach).")
    args = ap.parse_args()

    print(f"Opening PBF with pyrosm: {args.pbf}")
    osm = OSM(args.pbf)

    nodes_gdf, edges_gdf = osm.get_network(nodes=True, network_type="walking")
    if nodes_gdf is None or edges_gdf is None or len(nodes_gdf) == 0 or len(edges_gdf) == 0:
        print("Brak danych sieci 'walking' w podanym PBF.")
        return

    if nodes_gdf.crs is None:
        nodes_gdf.set_crs(epsg=4326, inplace=True)
    if edges_gdf.crs is None:
        edges_gdf.set_crs(epsg=4326, inplace=True)

    edges_len = edges_gdf.to_crs(epsg=3857).copy()
    edges_gdf = edges_gdf.copy()
    edges_gdf["length_m"] = edges_len.geometry.length

    if "id" in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"id": "node_id"})
    nodes_gdf["lon"] = nodes_gdf.geometry.x
    nodes_gdf["lat"] = nodes_gdf.geometry.y

    top_hw_before = (
        edges_gdf["highway"].astype(str).str.lower().value_counts().head(20).to_dict()
        if "highway" in edges_gdf.columns else {}
    )
    print("Top 'highway' BEFORE cleaning:", top_hw_before)

    edges_clean = clean_walkable_edges(edges_gdf)

    top_hw_after = (
        edges_clean["highway"].astype(str).str.lower().value_counts().head(20).to_dict()
        if "highway" in edges_clean.columns else {}
    )
    print("Top 'highway' AFTER cleaning:", top_hw_after)
    print(f"Edges: before={len(edges_gdf)}  after={len(edges_clean)}  (kept {len(edges_clean)/max(len(edges_gdf),1):.1%})")

    if "u" not in edges_clean.columns or "v" not in edges_clean.columns:
        print("Brak kolumn 'u'/'v' – mapuję końce geometrii do najbliższych węzłów (KDTree, EPSG:3857)…")
        nodes_3857 = nodes_gdf.to_crs(epsg=3857).copy()
        node_coords = np.vstack([nodes_3857.geometry.x.values, nodes_3857.geometry.y.values]).T
        node_ids = nodes_gdf["node_id"].values
        tree = cKDTree(node_coords)

        us, vs = [], []
        edges_3857 = edges_clean.to_crs(epsg=3857)
        for geom in edges_3857.geometry:
            coords = _safe_wkt_coords(geom)
            if not coords or len(coords) < 2:
                us.append(-1); vs.append(-1); continue
            p0 = coords[0]; p1 = coords[-1]
            _, i0 = tree.query([p0[0], p0[1]], k=1)
            _, i1 = tree.query([p1[0], p1[1]], k=1)
            us.append(int(node_ids[i0])); vs.append(int(node_ids[i1]))
        edges_clean["u"] = us
        edges_clean["v"] = vs

    nodes_out = nodes_gdf[["node_id", "lon", "lat"]].copy()

    edges_out = pd.DataFrame({
        "way_id": edges_clean.get("osmid", edges_clean.get("id", edges_clean.index)),
        "u": edges_clean["u"].astype(int),
        "v": edges_clean["v"].astype(int),
        "length_m": edges_clean["length_m"].astype(float),
        "highway": edges_clean.get("highway"),
        "surface": edges_clean.get("surface"),
        "oneway": edges_clean.get("oneway"),
    })

    edges_out["geometry_wkt"] = edges_clean.geometry.apply(lambda g: g.wkt if g is not None else None)

    nodes_csv = f"{args.out_prefix}_nodes.csv"
    edges_csv = f"{args.out_prefix}_edges.csv"
    nodes_out.to_csv(nodes_csv, index=False)
    edges_out.to_csv(edges_csv, index=False)
    print(f"Saved: {nodes_csv} ({len(nodes_out)})")
    print(f"Saved: {edges_csv} ({len(edges_out)})")

    adj = build_adjacency(edges_out)
    with open(f"{args.out_prefix}_adjacency.json", "w", encoding="utf8") as f:
        json.dump({str(k): v for k, v in adj.items()}, f, ensure_ascii=False)
    print(f"Saved: {args.out_prefix}_adjacency.json  (nodes with edges: {len(adj)})")

    if not args.no_plot:
        try:
            plt.figure(figsize=(9, 9))
            ax = plt.gca()

            unique_hw = edges_out["highway"].dropna().astype(str).str.lower().unique()
            cmap = plt.get_cmap("tab20")
            color_map = {hw: cmap(i % 20) for i, hw in enumerate(unique_hw)}

            for _, row in edges_clean.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                try:
                    coords = list(geom.coords)
                except Exception:
                    continue
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                hw = str(row.get("highway") or "").lower()
                ax.plot(xs, ys, linewidth=0.8, color=color_map.get(hw, (0.5, 0.5, 0.5)))

            ax.scatter(nodes_out["lon"], nodes_out["lat"], s=6, color="k", alpha=0.35)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Walking network (cleaned) from: {os.path.basename(args.pbf)}")
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
            out_png = f"{args.out_prefix}_plot.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {out_png}")
        except Exception as e:
            print("Błąd podczas rysowania:", e)


if __name__ == "__main__":
    main()
