
# """
# Użycie:
#   python graph_construction.py --pbf data/gdansk_5km.pbf --out-prefix walk_clean
#   python scripts/graph_construction.py --pbf data/gdynia_5km.pbf --out-prefix test --no-plot
# """
# import argparse
# import json
# import os
# from collections import defaultdict

# import matplotlib

# if "DISPLAY" not in os.environ:
#     matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from pyproj import Transformer
# from pyrosm import OSM
# from scipy.spatial import cKDTree
# from shapely import wkt

# def build_adjacency(edges_df, walk_speed_m_s=1.4):
#     adj = defaultdict(list)
#     for idx, row in edges_df.iterrows():
#         try:
#             u = int(row["u"]); v = int(row["v"])
#         except Exception:
#             continue
#         length = float(row.get("length_m", 0.0)) if row.get("length_m") is not None else 0.0
#         time_s = length / walk_speed_m_s if length and walk_speed_m_s else None
#         edge_id = int(row.get("way_id", idx))
#         attr = {
#             "edge_id": edge_id,
#             "length_m": length,
#             "cost_s": time_s,
#             "highway": row.get("highway"),
#         }
#         adj[u].append({"to": v, **attr})
#         oneway = str(row.get("oneway")).lower()
#         if oneway not in ("yes", "true", "1"):
#             adj[v].append({"to": u, **attr})
#     return adj


# def _safe_wkt_coords(geom):
#     if geom is None:
#         return None
#     try:
#         return list(geom.coords)
#     except Exception:
#         try:
#             g = wkt.loads(geom)
#             return list(g.coords)
#         except Exception:
#             return None


# def _norm(s):
#     return str(s).strip().lower() if s is not None else ""


# def clean_walkable_edges(edges_gdf):
#     """Zastosuj reguły 'walkable' opisane w rozmowie."""
#     edges = edges_gdf.copy()

#     for col in ("highway", "foot", "sidewalk", "motorroad", "oneway"):
#         if col in edges.columns:
#             edges[col + "_n"] = edges[col].map(_norm)
#         else:
#             edges[col + "_n"] = ""

#     always_walkable = {
#         "footway", "path", "pedestrian", "steps", "platform", "crossing",
#         "living_street", "cycleway", "track", "residential", "unclassified", "service",
#     }

#     big_roads = {"primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link"}

#     has_sidewalk = edges["sidewalk_n"].isin({"yes", "both", "left", "right"})
#     has_foot_ok = edges["foot_n"].isin({"yes", "designated", "permissive"})

#     is_fast = edges["highway_n"].isin({"motorway", "motorway_link", "trunk", "trunk_link"}) | (edges["motorroad_n"] == "yes")

#     walkable_mask = (
#         edges["highway_n"].isin(always_walkable)
#         | (edges["highway_n"].isin(big_roads) & (has_sidewalk | has_foot_ok))
#     )

#     cleaned = edges[walkable_mask & (~is_fast)].copy()
#     return cleaned


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--pbf", required=True, help="Ścieżka do .osm.pbf (najlepiej już przefiltrowanego osmium).")
#     ap.add_argument("--out-prefix", default="walk_clean", help="prefix plików wyjściowych CSV/JSON/PNG")
#     ap.add_argument("--no-plot", action="store_true", help="Pomiń rysowanie (szybciej na serwerach).")
#     args = ap.parse_args()

#     print(f"Opening PBF with pyrosm: {args.pbf}")
#     osm = OSM(args.pbf)

#     nodes_gdf, edges_gdf = osm.get_network(nodes=True, network_type="walking")
#     if nodes_gdf is None or edges_gdf is None or len(nodes_gdf) == 0 or len(edges_gdf) == 0:
#         print("Brak danych sieci 'walking' w podanym PBF.")
#         return

#     if nodes_gdf.crs is None:
#         nodes_gdf.set_crs(epsg=4326, inplace=True)
#     if edges_gdf.crs is None:
#         edges_gdf.set_crs(epsg=4326, inplace=True)

#     edges_len = edges_gdf.to_crs(epsg=3857).copy()
#     edges_gdf = edges_gdf.copy()
#     edges_gdf["length_m"] = edges_len.geometry.length

#     if "id" in nodes_gdf.columns:
#         nodes_gdf = nodes_gdf.rename(columns={"id": "node_id"})
#     nodes_gdf["lon"] = nodes_gdf.geometry.x
#     nodes_gdf["lat"] = nodes_gdf.geometry.y

#     top_hw_before = (
#         edges_gdf["highway"].astype(str).str.lower().value_counts().head(20).to_dict()
#         if "highway" in edges_gdf.columns else {}
#     )
#     print("Top 'highway' BEFORE cleaning:", top_hw_before)

#     edges_clean = clean_walkable_edges(edges_gdf)

#     top_hw_after = (
#         edges_clean["highway"].astype(str).str.lower().value_counts().head(20).to_dict()
#         if "highway" in edges_clean.columns else {}
#     )
#     print("Top 'highway' AFTER cleaning:", top_hw_after)
#     print(f"Edges: before={len(edges_gdf)}  after={len(edges_clean)}  (kept {len(edges_clean)/max(len(edges_gdf),1):.1%})")

#     if "u" not in edges_clean.columns or "v" not in edges_clean.columns:
#         print("Brak kolumn 'u'/'v' – mapuję końce geometrii do najbliższych węzłów (KDTree, EPSG:3857)…")
#         nodes_3857 = nodes_gdf.to_crs(epsg=3857).copy()
#         node_coords = np.vstack([nodes_3857.geometry.x.values, nodes_3857.geometry.y.values]).T
#         node_ids = nodes_gdf["node_id"].values
#         tree = cKDTree(node_coords)

#         us, vs = [], []
#         edges_3857 = edges_clean.to_crs(epsg=3857)
#         for geom in edges_3857.geometry:
#             coords = _safe_wkt_coords(geom)
#             if not coords or len(coords) < 2:
#                 us.append(-1); vs.append(-1); continue
#             p0 = coords[0]; p1 = coords[-1]
#             _, i0 = tree.query([p0[0], p0[1]], k=1)
#             _, i1 = tree.query([p1[0], p1[1]], k=1)
#             us.append(int(node_ids[i0])); vs.append(int(node_ids[i1]))
#         edges_clean["u"] = us
#         edges_clean["v"] = vs

#     nodes_out = nodes_gdf[["node_id", "lon", "lat"]].copy()

#     edges_out = pd.DataFrame({
#         "way_id": edges_clean.get("osmid", edges_clean.get("id", edges_clean.index)),
#         "u": edges_clean["u"].astype(int),
#         "v": edges_clean["v"].astype(int),
#         "length_m": edges_clean["length_m"].astype(float),
#         "highway": edges_clean.get("highway"),
#         "surface": edges_clean.get("surface"),
#         "oneway": edges_clean.get("oneway"),
#     })

#     edges_out["geometry_wkt"] = edges_clean.geometry.apply(lambda g: g.wkt if g is not None else None)

#     nodes_csv = f"{args.out_prefix}_nodes.csv"
#     edges_csv = f"{args.out_prefix}_edges.csv"
#     nodes_out.to_csv(nodes_csv, index=False)
#     edges_out.to_csv(edges_csv, index=False)
#     print(f"Saved: {nodes_csv} ({len(nodes_out)})")
#     print(f"Saved: {edges_csv} ({len(edges_out)})")

#     adj = build_adjacency(edges_out)
#     with open(f"{args.out_prefix}_adjacency.json", "w", encoding="utf8") as f:
#         json.dump({str(k): v for k, v in adj.items()}, f, ensure_ascii=False)
#     print(f"Saved: {args.out_prefix}_adjacency.json  (nodes with edges: {len(adj)})")

#     if not args.no_plot:
#         try:
#             plt.figure(figsize=(9, 9))
#             ax = plt.gca()

#             unique_hw = edges_out["highway"].dropna().astype(str).str.lower().unique()
#             cmap = plt.get_cmap("tab20")
#             color_map = {hw: cmap(i % 20) for i, hw in enumerate(unique_hw)}

#             for _, row in edges_clean.iterrows():
#                 geom = row.geometry
#                 if geom is None:
#                     continue
#                 try:
#                     coords = list(geom.coords)
#                 except Exception:
#                     continue
#                 xs = [c[0] for c in coords]
#                 ys = [c[1] for c in coords]
#                 hw = str(row.get("highway") or "").lower()
#                 ax.plot(xs, ys, linewidth=0.8, color=color_map.get(hw, (0.5, 0.5, 0.5)))

#             ax.scatter(nodes_out["lon"], nodes_out["lat"], s=6, color="k", alpha=0.35)
#             ax.set_aspect("equal", adjustable="box")
#             ax.set_title(f"Walking network (cleaned) from: {os.path.basename(args.pbf)}")
#             ax.set_xlabel("lon"); ax.set_ylabel("lat")
#             out_png = f"{args.out_prefix}_plot.png"
#             plt.savefig(out_png, dpi=150, bbox_inches="tight")
#             plt.close()
#             print(f"Saved plot: {out_png}")
#         except Exception as e:
#             print("Błąd podczas rysowania:", e)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
graph_construction.py

Build a cleaned walking graph from a PBF extract and optionally snap POIs
to the nearest walkable node. Outputs SNAP results as JSON (no Parquet).
This version removes any interactive plotting and only saves two PNGs:
 - {out_prefix}_plot.png        (network plot)
 - {out_prefix}_pois_snapped_diag.png  (POI -> node diagnostic)
"""
import argparse
import json
import os
from collections import defaultdict

# Force matplotlib backend in headless environments
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

# ------------------------- utility functions -------------------------
def _norm(s):
    return str(s).strip().lower() if s is not None else ""

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

def parse_poi_filters(s: str):
    """Parse a comma-separated list like 'amenity=pharmacy,shop=supermarket'"""
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append((p.strip(), None))
    return out

# ------------------------- cleaning policy -------------------------
def clean_walkable_edges(edges_gdf):
    """
    Filter edges to keep pedestrian-accessible ways (heuristic).
    """
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

# ------------------------- snap POIs to walkable nodes -------------------------
def snap_pois_to_walkable_nodes(osm, nodes_gdf, edges_clean, out_prefix,
                                poi_filters=None, max_snap_distance=None,
                                diag_plot=True, max_lines_plot=200):
    """
    Snap POIs to nearest walkable nodes. Returns (pois_out_list, node_to_poiids, stats).
    """
    # Build set of walkable node ids from edges_clean.u/v
    walkable_node_ids = set()
    if "u" in edges_clean.columns and "v" in edges_clean.columns:
        try:
            u_vals = edges_clean["u"].dropna().astype(np.int64)
            v_vals = edges_clean["v"].dropna().astype(np.int64)
            walkable_node_ids.update(map(int, u_vals.to_numpy()))
            walkable_node_ids.update(map(int, v_vals.to_numpy()))
        except Exception:
            for v in edges_clean["u"].dropna().tolist():
                try: walkable_node_ids.add(int(v))
                except Exception: pass
            for v in edges_clean["v"].dropna().tolist():
                try: walkable_node_ids.add(int(v))
                except Exception: pass

    if not walkable_node_ids:
        nodes_for_tree = nodes_gdf.copy()
    else:
        nodes_for_tree = nodes_gdf[nodes_gdf["node_id"].isin(walkable_node_ids)].copy()
        if len(nodes_for_tree) == 0:
            nodes_for_tree = nodes_gdf.copy()

    # Build KDTree in EPSG:3857
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    nodes_proj = nodes_for_tree.to_crs(epsg=3857).copy()
    node_coords = np.vstack([nodes_proj.geometry.x.values, nodes_proj.geometry.y.values]).T.astype(np.float32)
    node_ids = nodes_for_tree["node_id"].to_numpy(dtype=np.int64)
    tree = cKDTree(node_coords)

    # get POIs via pyrosm
    custom = {}
    if poi_filters:
        for k, v in poi_filters:
            if v is not None:
                custom.setdefault(k, []).append(v)
    try:
        pois_gdf = osm.get_pois(custom_filter=custom) if custom else osm.get_pois()
    except Exception:
        pois_gdf = osm.get_pois()

    if pois_gdf is None or len(pois_gdf) == 0:
        print("No POIs found in this tile.")
        return None, None, None

    # post-filter for keys without explicit values
    if poi_filters:
        mask = pd.Series([True] * len(pois_gdf), index=pois_gdf.index)
        for k, v in poi_filters:
            if v is None:
                if k in pois_gdf.columns:
                    mask &= pois_gdf[k].notna()
                else:
                    if "tags" in pois_gdf.columns:
                        mask &= pois_gdf["tags"].astype(str).str.contains(fr"'{k}'|\"{k}\"|{k}:", na=False)
                    else:
                        mask &= False
            else:
                if k in pois_gdf.columns:
                    mask &= pois_gdf[k].astype(str).str.lower() == v.lower()
                else:
                    if "tags" in pois_gdf.columns:
                        mask &= pois_gdf["tags"].astype(str).str.contains(fr"{k}.*{v}", case=False, na=False)
                    else:
                        mask &= False
        pois_gdf = pois_gdf[mask].copy()

    if len(pois_gdf) == 0:
        print("No POIs after post-filtering.")
        return None, None, None

    # ensure POI lon/lat using representative_point for polygons
    if "geometry" in pois_gdf.columns:
        try:
            rep_pts = pois_gdf.geometry.representative_point()
            pois_lon = rep_pts.x.to_numpy()
            pois_lat = rep_pts.y.to_numpy()
        except Exception:
            rep_pts = pois_gdf.geometry.apply(lambda g: (g.representative_point() if g is not None else None))
            pois_lon = np.array([p.x if p is not None else np.nan for p in rep_pts])
            pois_lat = np.array([p.y if p is not None else np.nan for p in rep_pts])
    else:
        if "lon" in pois_gdf.columns and "lat" in pois_gdf.columns:
            pois_lon = pois_gdf["lon"].to_numpy()
            pois_lat = pois_gdf["lat"].to_numpy()
        else:
            print("POIs lack geometry and lon/lat -> cannot snap.")
            return None, None, None

    # project POIs to metric coords
    px, py = transformer.transform(pois_lon, pois_lat)
    poi_coords = np.vstack([px, py]).T.astype(np.float32)

    # query nearest
    print("Querying KDTree for nearest walkable nodes...")
    dists, idxs = tree.query(poi_coords, k=1)
    dists = dists.astype(float)
    idxs = idxs.astype(np.int64)
    nearest_node_ids = node_ids[idxs]

    # ensure poi id column
    poi_id_col = None
    for c in ("id", "osm_id", "osmid"):
        if c in pois_gdf.columns:
            poi_id_col = c; break
    if poi_id_col is None:
        pois_gdf = pois_gdf.reset_index(drop=True)
        poi_id_col = "__poi_id__"
        pois_gdf[poi_id_col] = np.arange(len(pois_gdf))

    # build list of dicts to save as JSON
    pois_out_list = []
    for i, row in pois_gdf.iterrows():
        pid = int(row[poi_id_col]) if poi_id_col in row else int(i)
        name = row.get("name") if "name" in row else None
        lon = float(pois_lon[i]) if not np.isnan(pois_lon[i]) else None
        lat = float(pois_lat[i]) if not np.isnan(pois_lat[i]) else None
        nearest_node = int(nearest_node_ids[i])
        dist_m = float(dists[i])
        off_net = bool(dist_m > float(max_snap_distance)) if max_snap_distance is not None else False
        rec = {
            "poi_id": pid,
            "name": name,
            "lon": lon,
            "lat": lat,
            "nearest_node_id": nearest_node,
            "dist_m_to_node": dist_m,
            "off_network": off_net,
        }
        if "tags" in row and row["tags"] is not None:
            rec["tags"] = row["tags"]
        for c in ("amenity","shop","leisure","tourism","category"):
            if c in row and row[c] is not None:
                rec[c] = row[c]
        pois_out_list.append(rec)

    # save POIs JSON array
    pois_out_path = f"{out_prefix}_pois_snapped.json"
    with open(pois_out_path, "w", encoding="utf8") as fh:
        json.dump(pois_out_list, fh, ensure_ascii=False)
    print("Saved snapped POIs JSON:", pois_out_path)

    # build reverse mapping node_id -> [poi_id,...]
    node_to_poiids = defaultdict(list)
    for rec in pois_out_list:
        node_to_poiids[str(rec["nearest_node_id"])].append(rec["poi_id"])
    node_map_path = f"{out_prefix}_node_pois.json"
    with open(node_map_path, "w", encoding="utf8") as fh:
        json.dump(node_to_poiids, fh, ensure_ascii=False)
    print("Saved node->POIs mapping JSON:", node_map_path)

    # stats
    dseries = np.array([r["dist_m_to_node"] for r in pois_out_list], dtype=float)
    stats = {
        "count": int(len(dseries)),
        "mean_m": float(dseries.mean()) if len(dseries) else None,
        "median_m": float(np.median(dseries)) if len(dseries) else None,
        "max_m": float(dseries.max()) if len(dseries) else None,
    }
    if max_snap_distance is not None and len(dseries):
        stats["pct_over_threshold"] = float((dseries > float(max_snap_distance)).sum()) / len(dseries) * 100.0
    print("POI->node snap stats:", stats)

    # Diagnostic plot (save only PNG)
    if diag_plot:
        try:
            fig, ax = plt.subplots(figsize=(9,9))
            ax.scatter(node_coords[:,0], node_coords[:,1], s=4, color="gray", alpha=0.5)
            ax.scatter(poi_coords[:,0], poi_coords[:,1], s=10, color="red", alpha=0.9)
            nlines = min(len(poi_coords), max_lines_plot)
            for i in range(nlines):
                x0,y0 = poi_coords[i]
                kidx = idxs[i]
                x1,y1 = node_coords[int(kidx)]
                ax.plot([x0,x1],[y0,y1], linewidth=0.6, color="blue", alpha=0.6)
            ax.set_title("POI (red) -> nearest walkable node (gray) (EPSG:3857)")
            out_png = f"{out_prefix}_pois_snapped_diag.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Saved diagnostic plot:", out_png)
        except Exception as e:
            print("Diagnostic plot failed:", e)

    return pois_out_list, node_to_poiids, stats

# ------------------------- main pipeline -------------------------
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
        attr = {"edge_id": edge_id, "length_m": length, "cost_s": time_s, "highway": row.get("highway")}
        adj[u].append({"to": v, **attr})
        oneway = str(row.get("oneway")).lower()
        if oneway not in ("yes", "true", "1"):
            adj[v].append({"to": u, **attr})
    return adj

def save_network_png(edges_clean, nodes_out, out_prefix):
    """Save the walking network plot as PNG (lon/lat coordinates)."""
    try:
        fig, ax = plt.subplots(figsize=(9,9))
        unique_hw = edges_clean["highway"].dropna().astype(str).str.lower().unique() if "highway" in edges_clean.columns else []
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
            ax.plot(xs, ys, linewidth=0.6, color=color_map.get(hw, (0.5,0.5,0.5)))
        ax.scatter(nodes_out["lon"], nodes_out["lat"], s=6, color="k", alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Walking network (cleaned) from: {os.path.basename(out_prefix)}")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        out_png = f"{out_prefix}_plot.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved network PNG:", out_png)
    except Exception as e:
        print("Network PNG generation failed:", e)

def main():
    ap = argparse.ArgumentParser(description="Build cleaned walking graph and optionally snap POIs to nodes (JSON outputs)."
    )
    ap.add_argument("--pbf", required=True, help="Path to tile .pbf")
    ap.add_argument("--out-prefix", default="walk_clean", help="Output prefix")
    ap.add_argument("--no-plot", action="store_true", help="Skip plots (saves time)")
    ap.add_argument("--snap-pois", action="store_true", help="Snap POIs to nearest walkable node")
    ap.add_argument("--poi-filters", default="", help="Comma-separated filters for POIs: key=value or key")
    ap.add_argument("--max-snap-distance", type=float, default=None, help="Optional max snap distance in meters")
    args = ap.parse_args()

    print("Opening PBF with pyrosm:", args.pbf)
    osm = OSM(args.pbf)

    nodes_gdf, edges_gdf = osm.get_network(nodes=True, network_type="walking")
    if nodes_gdf is None or edges_gdf is None or len(nodes_gdf) == 0 or len(edges_gdf) == 0:
        print("No 'walking' network data in the provided PBF.")
        return

    if nodes_gdf.crs is None:
        nodes_gdf.set_crs(epsg=4326, inplace=True)
    if edges_gdf.crs is None:
        edges_gdf.set_crs(epsg=4326, inplace=True)

    edges_3857 = edges_gdf.to_crs(epsg=3857).copy()
    edges_gdf = edges_gdf.copy()
    edges_gdf["length_m"] = edges_3857.geometry.length

    if "id" in nodes_gdf.columns and "node_id" not in nodes_gdf.columns:
        nodes_gdf = nodes_gdf.rename(columns={"id": "node_id"})
    nodes_gdf["lon"] = nodes_gdf.geometry.x
    nodes_gdf["lat"] = nodes_gdf.geometry.y

    print("Sample highway types before cleaning:")
    if "highway" in edges_gdf.columns:
        print(edges_gdf["highway"].astype(str).str.lower().value_counts().head(20).to_dict())

    edges_clean = clean_walkable_edges(edges_gdf)

    print("Sample highway types after cleaning:")
    if "highway" in edges_clean.columns:
        print(edges_clean["highway"].astype(str).str.lower().value_counts().head(20).to_dict())
    print(f"Edges: before={len(edges_gdf)} after={len(edges_clean)} kept={(len(edges_clean)/max(len(edges_gdf),1)):.1%}")

    # map u/v if not present
    if "u" not in edges_clean.columns or "v" not in edges_clean.columns:
        print("u/v columns missing in edges_clean — mapping geometry endpoints to nearest nodes (KDTree).")
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

    # prepare nodes/edges CSVs
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

    # optional: snap POIs -> JSON outputs (and PNG diag saved inside)
    if args.snap_pois:
        filters = parse_poi_filters(args.poi_filters)
        pois_list, node_to_pois_map, snap_stats = snap_pois_to_walkable_nodes(
            osm, nodes_gdf, edges_clean, args.out_prefix,
            poi_filters=filters, max_snap_distance=args.max_snap_distance,
            diag_plot=not args.no_plot
        )
        if pois_list is None:
            print("No POIs found or snapping failed.")
        else:
            print("POI snapping completed. Stats:", snap_stats)

    # build adjacency and save
    adj = build_adjacency(edges_out)
    adj_path = f"{args.out_prefix}_adjacency.json"
    with open(adj_path, "w", encoding="utf8") as fh:
        json.dump({str(k): v for k, v in adj.items()}, fh, ensure_ascii=False)
    print("Saved adjacency:", adj_path, f"(nodes with edges: {len(adj)})")


    # Always save network PNG unless user asks to skip plots
    if not args.no_plot:
        save_network_png(edges_clean, nodes_out, args.out_prefix)

if __name__ == "__main__":
    main()



