#!/usr/bin/env python3
"""
sgraphs_to_json.py
------------------
Converts a s_graphs+ dump directory into a single Hydra DSG-style JSON.

Reads:
  graph.g2o                          → all pose graph nodes + edges
  keyframes/<id>/kf_data.txt         → keyframe poses + metadata
  x_vert_planes/<id>/x_plane_data.txt
  y_vert_planes/<id>/y_plane_data.txt → vertical plane geometry
  hort_planes/<id>/hort_plane_data.txt→ horizontal plane geometry
  walls/<id>/wall_data.txt           → wall semantics
  rooms/<id>/room_data.txt           → room semantics
  floors/<id>/floor_data.txt         → floor semantics
  anchor_node.txt                    → graph anchor pose
  session_details.txt                → metadata

Also subscribes (RELIABLE QoS) to:
  /s_graphs/graph_keyframes          → enriches keyframes with live poses
  /s_graphs/all_map_planes           → enriches planes with normals

Usage
-----
source ~/workspaces/s_graphs/install/setup.bash
python3 sgraphs_to_json.py --output ~/output/my_scene.json

Options
-------
--output PATH       Output JSON file (default: sgraphs_export_<timestamp>.json)
--dump-dir PATH     Directory for s_graphs dump files (default: /tmp/sgraphs_dump)
--from-dir PATH     Skip the service call entirely — read directly from an existing
                    session directory, e.g.:
                    --from-dir /tmp/sgraphs_dump/s_graphs_data_2026-4_21_10_35_41
--no-wait           Skip waiting for live topic data; call dump immediately
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from situational_graphs_msgs.srv import DumpGraph
from situational_graphs_msgs.msg import PlanesData
from situational_graphs_reasoning_msgs.msg import GraphKeyframes


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(v):
    try:
        return float(v)
    except Exception:
        return v

def quaternion_to_dict(q):
    return {"x": q.x, "y": q.y, "z": q.z, "w": q.w}

def vector3_to_list(v):
    return [v.x, v.y, v.z]

def pose_to_dict(p):
    return {
        "position": [p.position.x, p.position.y, p.position.z],
        "orientation": quaternion_to_dict(p.orientation),
    }

def stamp_to_dict(s):
    return {"sec": s.sec, "nanosec": s.nanosec}

def infer_layer(node_type: str) -> int:
    mapping = {"keyframe": 1, "floor": 2, "room": 3, "wall": 4, "plane": 5}
    for key, layer in mapping.items():
        if key.lower() in node_type.lower():
            return layer
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Dump file parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_kv_txt(path: Path) -> dict:
    """
    Parse a key: value text file (used by kf_data, room_data, wall_data, etc.)
    Handles both   'key: value'   and   'key value'   formats.
    Multi-value lines become lists.
    """
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try colon-separated first
            if ":" in line:
                key, _, rest = line.partition(":")
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, rest = parts
            key = key.strip()
            values = rest.strip().split()
            if len(values) == 1:
                result[key] = _f(values[0])
            else:
                result[key] = [_f(v) for v in values]
    return result


def parse_g2o(path: Path) -> tuple[list, list]:
    """
    Parse a .g2o file into (vertices, edges).

    Supported vertex types:
      VERTEX_SE3:QUAT   id x y z qx qy qz qw
      VERTEX_PLANE      id nx ny nz d
      VERTEX_ROOM       id x y
      VERTEX_FLOOR      id x y z

    Supported edge types (all remaining fields → information matrix):
      EDGE_SE3:QUAT     from to  x y z qx qy qz qw  <9 info values>
      EDGE_SE3_PLANE    from to  <measurement>  <info>
      EDGE_PLANE_ROOM   from to  ...
      EDGE_ROOM_FLOOR   from to  ...
      (any EDGE_* line)
    """
    vertices = []
    edges = []

    if not path.exists():
        return vertices, edges

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]

            if tag.startswith("VERTEX"):
                v_id = int(parts[1])
                floats = [float(x) for x in parts[2:]]

                if "SE3" in tag:
                    v = {
                        "id": v_id,
                        "type": "keyframe",
                        "layer": 1,
                        "pose": {
                            "position": floats[0:3],
                            "orientation": {
                                "x": floats[3], "y": floats[4],
                                "z": floats[5], "w": floats[6],
                            },
                        },
                    }
                elif "PLANE" in tag:
                    v = {
                        "id": v_id,
                        "type": "plane",
                        "layer": 5,
                        "normal": {
                            "nx": floats[0], "ny": floats[1],
                            "nz": floats[2], "d": floats[3],
                        },
                    }
                elif "ROOM" in tag:
                    v = {
                        "id": v_id,
                        "type": "room",
                        "layer": 3,
                        "position": floats,
                    }
                elif "FLOOR" in tag:
                    v = {
                        "id": v_id,
                        "type": "floor",
                        "layer": 2,
                        "position": floats,
                    }
                else:
                    v = {"id": v_id, "type": tag, "layer": 0, "data": floats}

                vertices.append(v)

            elif tag.startswith("EDGE"):
                src = int(parts[1])
                tgt = int(parts[2])
                floats = [float(x) for x in parts[3:]]
                edges.append({
                    "source": src,
                    "target": tgt,
                    "type": tag,
                    "measurement": floats[:7] if len(floats) >= 7 else floats,
                    "information": floats[7:] if len(floats) > 7 else [],
                })

    return vertices, edges


def parse_plane_csv(path: Path) -> list:
    """Parse the summary plane_data.csv files."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _f(v) for k, v in row.items()})
    return rows


def scan_subdir_txts(subdir: Path, txt_name: str) -> dict:
    """
    Walk subdir/<id>/<txt_name> and return {id: parsed_dict}.
    id is inferred from the subdirectory name.
    """
    result = {}
    if not subdir.exists():
        return result
    for child in sorted(subdir.iterdir()):
        if child.is_dir():
            txt = child / txt_name
            if txt.exists():
                data = parse_kv_txt(txt)
                data["_dump_id"] = int(child.name) if child.name.isdigit() else child.name
                result[child.name] = data
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ─────────────────────────────────────────────────────────────────────────────

class SGrapsToJsonNode(Node):

    def __init__(self, output_path: Path, dump_dir: Path, no_wait: bool,
                 from_dir: Path = None):
        super().__init__("sgraphs_to_json")

        self.output_path = output_path
        self.dump_dir = dump_dir
        self.no_wait = no_wait
        self.from_dir = from_dir

        self._latest_keyframes = None
        self._latest_planes = None

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.create_subscription(GraphKeyframes, "/s_graphs/graph_keyframes",
                                 self._keyframes_cb, reliable_qos)
        self.create_subscription(PlanesData, "/s_graphs/all_map_planes",
                                 self._planes_cb, best_effort_qos)

        # --from-dir: skip service call, read existing dump immediately
        if from_dir is not None:
            if not from_dir.exists():
                self.get_logger().error(f"--from-dir path does not exist: {from_dir}")
                sys.exit(1)
            self.get_logger().info(f"--from-dir mode: reading {from_dir}")
            self.get_logger().info("Waiting 3s for any live topic data...")
            self._from_dir_timer = self.create_timer(3.0, self._from_dir_ready)
            return

        self._dump_client = self.create_client(DumpGraph, "/s_graphs/dump")
        self.get_logger().info("Waiting for /s_graphs/dump service...")
        if not self._dump_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("Service not available. Is s_graphs_node running?")
            sys.exit(1)

        if no_wait:
            self._trigger_dump()
        else:
            self.get_logger().info("Waiting up to 15s for keyframes + planes...")
            self._wait_elapsed = 0
            self._wait_timer = self.create_timer(1.0, self._check_ready)

    def _keyframes_cb(self, msg):
        if self._latest_keyframes is None:
            self.get_logger().info(f"Got keyframes: {len(msg.keyframes)}")
        self._latest_keyframes = msg

    def _planes_cb(self, msg):
        if self._latest_planes is None:
            self.get_logger().info(
                f"Got planes: {len(msg.x_planes)} x-planes, {len(msg.y_planes)} y-planes"
            )
        self._latest_planes = msg

    def _check_ready(self):
        self._wait_elapsed += 1
        have_kf = self._latest_keyframes is not None
        have_pl = self._latest_planes is not None
        if have_kf and have_pl:
            self._wait_timer.cancel()
            self.get_logger().info("Ready — dumping.")
            self._trigger_dump()
        elif self._wait_elapsed >= 15:
            self._wait_timer.cancel()
            self.get_logger().warn(
                f"Timeout (keyframes={'yes' if have_kf else 'NO'}, "
                f"planes={'yes' if have_pl else 'NO'}) — dumping anyway."
            )
            self._trigger_dump()

    def _from_dir_ready(self):
        self._from_dir_timer.cancel()
        self.get_logger().info(f"Converting from existing dump: {self.from_dir}")
        try:
            output = self._build_json(self.from_dir)
            self._write_json(output)
        except Exception as e:
            self.get_logger().error(f"Conversion failed: {e}")
            import traceback; traceback.print_exc()
        rclpy.shutdown()

    def _trigger_dump(self):
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        req = DumpGraph.Request()
        req.destination = str(self.dump_dir)
        self.get_logger().info(f"Calling /s_graphs/dump → {self.dump_dir}")
        future = self._dump_client.call_async(req)
        future.add_done_callback(self._dump_done)

    def _dump_done(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Dump call failed: {e}")
            rclpy.shutdown()
            return

        if not response.success:
            self.get_logger().error("Dump returned success=False")
            rclpy.shutdown()
            return

        # Give s_graphs a moment to finish writing
        time.sleep(1.0)

        # Find the timestamped subdirectory s_graphs created
        subdirs = sorted(
            [d for d in self.dump_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not subdirs:
            self.get_logger().error(f"No subdirectories found in {self.dump_dir}")
            rclpy.shutdown()
            return

        session_dir = subdirs[0]
        self.get_logger().info(f"Reading dump from: {session_dir}")

        try:
            output = self._build_json(session_dir)
            self._write_json(output)
        except Exception as e:
            self.get_logger().error(f"Conversion failed: {e}")
            import traceback; traceback.print_exc()

        rclpy.shutdown()

    # ─────────────────────────────────────────────────────────────────────────
    # Build the JSON
    # ─────────────────────────────────────────────────────────────────────────

    def _build_json(self, session_dir: Path) -> dict:

        output = {
            "SPARK_DSG_header": {
                "project_name": "s_graphs_plus",
                "source_node": "s_graphs_node",
                "session_dir": str(session_dir),
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "version": {"major": 1, "minor": 0, "patch": 0},
            },
            "directed": True,
            "multigraph": False,
            "layer_names": {
                "KEYFRAMES": {"layer": 1, "partition": 0},
                "FLOORS":    {"layer": 2, "partition": 0},
                "ROOMS":     {"layer": 3, "partition": 0},
                "WALLS":     {"layer": 4, "partition": 0},
                "PLANES":    {"layer": 5, "partition": 0},
            },
            "nodes": [],
            "edges": [],
            "keyframes": [],
            "rooms": [],
            "walls": [],
            "floors": [],
            "planes": {"x_planes": [], "y_planes": [], "hort_planes": []},
            "metadata": {},
        }

        node_ids_seen = set()

        def add_node(node: dict):
            if node["id"] not in node_ids_seen:
                output["nodes"].append(node)
                node_ids_seen.add(node["id"])

        # ── 1. g2o pose graph ─────────────────────────────────────────────
        g2o_vertices, g2o_edges = parse_g2o(session_dir / "graph.g2o")

        for v in g2o_vertices:
            add_node({
                "id": v["id"],
                "type": v["type"],
                "layer": v["layer"],
                "partition": 0,
                "attributes": {k: val for k, val in v.items()
                               if k not in ("id", "type", "layer")},
            })

        for e in g2o_edges:
            output["edges"].append({
                "source": e["source"],
                "target": e["target"],
                "info": {
                    "type": e["type"],
                    "weighted": True,
                    "weight": 1.0,
                    "measurement": e["measurement"],
                    "information": e["information"],
                },
            })

        self.get_logger().info(
            f"g2o: {len(g2o_vertices)} vertices, {len(g2o_edges)} edges"
        )

        # ── 2. Keyframes ──────────────────────────────────────────────────
        kf_dump = scan_subdir_txts(session_dir / "keyframes", "kf_data.txt")
        live_kf_map = {}
        if self._latest_keyframes is not None:
            for kf in self._latest_keyframes.keyframes:
                live_kf_map[kf.id] = kf

        for kf_id_str, kf_data in kf_dump.items():
            kf_id = int(kf_id_str)
            kf_entry = {
                "id": kf_id,
                "type": "keyframe",
                "dump_data": kf_data,
            }
            # Enrich with live subscription data if available
            if kf_id in live_kf_map:
                lkf = live_kf_map[kf_id]
                kf_entry["pose"] = pose_to_dict(lkf.pose)
                kf_entry["timestamp"] = stamp_to_dict(lkf.header.stamp)
                kf_entry["frame_id"] = lkf.header.frame_id
                kf_entry["pointcloud_fields"] = [f.name for f in lkf.pointcloud.fields]

            output["keyframes"].append(kf_entry)
            add_node({
                "id": kf_id,
                "type": "keyframe",
                "layer": 1,
                "partition": 0,
                "attributes": kf_data,
            })

        self.get_logger().info(f"keyframes: {len(output['keyframes'])}")

        # ── 3. Rooms ──────────────────────────────────────────────────────
        room_dump = scan_subdir_txts(session_dir / "rooms", "room_data.txt")
        for r_id_str, r_data in room_dump.items():
            output["rooms"].append({"id": r_id_str, **r_data})
            rid = int(r_id_str) if r_id_str.isdigit() else r_id_str
            add_node({
                "id": rid,
                "type": "room",
                "layer": 3,
                "partition": 0,
                "attributes": r_data,
            })

        self.get_logger().info(f"rooms: {len(output['rooms'])}")

        # ── 4. Walls ──────────────────────────────────────────────────────
        wall_dump = scan_subdir_txts(session_dir / "walls", "wall_data.txt")
        for w_id_str, w_data in wall_dump.items():
            output["walls"].append({"id": w_id_str, **w_data})
            wid = int(w_id_str) if w_id_str.isdigit() else w_id_str
            add_node({
                "id": wid,
                "type": "wall",
                "layer": 4,
                "partition": 0,
                "attributes": w_data,
            })

        self.get_logger().info(f"walls: {len(output['walls'])}")

        # ── 5. Floors ─────────────────────────────────────────────────────
        floor_dump = scan_subdir_txts(session_dir / "floors", "floor_data.txt")
        for f_id_str, f_data in floor_dump.items():
            output["floors"].append({"id": f_id_str, **f_data})
            fid = int(f_id_str) if f_id_str.isdigit() else f_id_str
            add_node({
                "id": fid,
                "type": "floor",
                "layer": 2,
                "partition": 0,
                "attributes": f_data,
            })

        self.get_logger().info(f"floors: {len(output['floors'])}")

        # ── 6. Vertical planes (x and y) ──────────────────────────────────
        x_plane_dump = scan_subdir_txts(session_dir / "x_vert_planes", "x_plane_data.txt")
        y_plane_dump = scan_subdir_txts(session_dir / "y_vert_planes", "y_plane_data.txt")
        hort_plane_dump = scan_subdir_txts(session_dir / "hort_planes", "hort_plane_data.txt")

        for p_id_str, p_data in x_plane_dump.items():
            output["planes"]["x_planes"].append({"id": p_id_str, **p_data})
            pid = int(p_id_str) if p_id_str.isdigit() else p_id_str
            add_node({"id": pid, "type": "plane_x", "layer": 5,
                      "partition": 0, "attributes": p_data})

        for p_id_str, p_data in y_plane_dump.items():
            output["planes"]["y_planes"].append({"id": p_id_str, **p_data})
            pid = int(p_id_str) if p_id_str.isdigit() else p_id_str
            add_node({"id": pid, "type": "plane_y", "layer": 5,
                      "partition": 0, "attributes": p_data})

        for p_id_str, p_data in hort_plane_dump.items():
            output["planes"]["hort_planes"].append({"id": p_id_str, **p_data})

        # Enrich with live plane subscription data
        if self._latest_planes is not None:
            pl = self._latest_planes
            output["planes"]["live_x_planes"] = [
                {"id": p.id,
                 "normal": {"nx": p.nx, "ny": p.ny, "nz": p.nz, "d": p.d},
                 "orientation": vector3_to_list(p.plane_orientation),
                 "points": [vector3_to_list(pt) for pt in p.plane_points],
                 "data_source": p.data_source}
                for p in pl.x_planes
            ]
            output["planes"]["live_y_planes"] = [
                {"id": p.id,
                 "normal": {"nx": p.nx, "ny": p.ny, "nz": p.nz, "d": p.d},
                 "orientation": vector3_to_list(p.plane_orientation),
                 "points": [vector3_to_list(pt) for pt in p.plane_points],
                 "data_source": p.data_source}
                for p in pl.y_planes
            ]

        self.get_logger().info(
            f"planes: {len(x_plane_dump)} x, {len(y_plane_dump)} y, "
            f"{len(hort_plane_dump)} hort"
        )

        # ── 7. Anchor node + session details ──────────────────────────────
        anchor = parse_kv_txt(session_dir / "anchor_node.txt")
        if anchor:
            output["metadata"]["anchor_node"] = anchor

        session = parse_kv_txt(session_dir / "session_details.txt")
        if session:
            output["metadata"]["session_details"] = session

        # ── 8. Summary CSV files ──────────────────────────────────────────
        for csv_path in session_dir.rglob("*.csv"):
            key = str(csv_path.relative_to(session_dir))
            output["metadata"][key] = parse_plane_csv(csv_path)

        return output

    # ─────────────────────────────────────────────────────────────────────────
    # Write output
    # ─────────────────────────────────────────────────────────────────────────

    def _write_json(self, data: dict):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)

        p = data["planes"]
        self.get_logger().info(
            f"\n{'─'*50}\n"
            f"  Output:    {self.output_path}\n"
            f"  Nodes:     {len(data['nodes'])}\n"
            f"  Edges:     {len(data['edges'])}\n"
            f"  Keyframes: {len(data['keyframes'])}\n"
            f"  Rooms:     {len(data['rooms'])}\n"
            f"  Walls:     {len(data['walls'])}\n"
            f"  Floors:    {len(data['floors'])}\n"
            f"  Planes:    {len(p['x_planes'])} x-vert, "
            f"{len(p['y_planes'])} y-vert, "
            f"{len(p['hort_planes'])} hort\n"
            f"{'─'*50}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert s_graphs+ dump to JSON")
    parser.add_argument(
        "--output", "-o",
        default=f"sgraphs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument("--dump-dir", default="/tmp/sgraphs_dump")
    parser.add_argument(
        "--from-dir",
        default=None,
        help="Read directly from an existing session directory, skipping the dump service. "
             "e.g. --from-dir /tmp/sgraphs_dump/s_graphs_data_2026-4_21_10_35_41"
    )
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = SGrapsToJsonNode(
        output_path=Path(args.output),
        dump_dir=Path(args.dump_dir),
        no_wait=args.no_wait,
        from_dir=Path(args.from_dir) if args.from_dir else None,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()