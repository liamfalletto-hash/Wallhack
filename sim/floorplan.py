"""Floor plan loading, wall geometry and route planning for the simulator."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
MATERIALS = frozenset({"drywall", "wood", "glass", "brick", "concrete"})
BODY_RADIUS_M = 0.25  # people keep at least this far from walls and furniture
NODE_OFFSET_M = 0.35  # how far route nodes sit from obstacle corners
DOOR_APPROACH_M = 0.5  # route nodes on each side of a doorway
EPS = 1e-9


@dataclass(frozen=True, eq=False)
class FloorPlan:
    name: str
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    wall_segs: np.ndarray  # (N, 2, 2)
    wall_materials: tuple[str, ...]
    obstacle_polys: tuple[np.ndarray, ...]
    rooms: dict[str, np.ndarray]
    blocking_segs: np.ndarray  # walls plus furniture edges, (M, 2, 2)
    nav_nodes: np.ndarray  # (K, 2) candidate route points, already clear of everything


def load(path: str | Path) -> FloorPlan:
    return parse(json.loads(Path(path).read_text()))


def parse(d: dict) -> FloorPlan:
    if d.get("v") != FORMAT_VERSION:
        raise ValueError(f"unsupported floor plan version {d.get('v')!r}, expected {FORMAT_VERSION}")
    if d.get("units", "m") != "m":
        raise ValueError("floor plan units must be metres")
    b = d["bounds"]
    bounds = (float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"]))

    walls = d.get("walls", [])
    if not walls:
        raise ValueError("floor plan has no walls")
    segs = np.array([[w["a"], w["b"]] for w in walls], dtype=float)
    if np.any(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) < EPS):
        raise ValueError("floor plan has a zero-length wall")
    materials = tuple(w.get("material", "drywall") for w in walls)
    unknown = set(materials) - MATERIALS
    if unknown:
        raise ValueError(f"unknown wall materials {sorted(unknown)}, allowed: {sorted(MATERIALS)}")

    polys = tuple(_parse_poly(o["polygon"], o.get("name", "obstacle")) for o in d.get("obstacles", []))
    rooms = {r["name"]: _parse_poly(r["polygon"], r["name"]) for r in d.get("rooms", [])}
    doors = np.array([[dr["a"], dr["b"]] for dr in d.get("doors", [])], dtype=float).reshape(-1, 2, 2)

    blocking = np.concatenate([segs, *(_poly_edges(p) for p in polys)])
    candidates = np.concatenate([_door_nodes(doors), *(_corner_nodes(p) for p in polys)]).reshape(-1, 2)
    fp = FloorPlan(d.get("name", "unnamed"), bounds, segs, materials, polys, rooms, blocking, np.empty((0, 2)))
    keep = [i for i, n in enumerate(candidates) if is_free_point(fp, n)]
    return FloorPlan(fp.name, bounds, segs, materials, polys, rooms, blocking, candidates[keep])


def crossed_walls(fp: FloorPlan, p: np.ndarray, q: np.ndarray) -> list[str]:
    hits = _segments_intersect(np.asarray(p, float), np.asarray(q, float), fp.wall_segs)
    return [m for m, h in zip(fp.wall_materials, hits, strict=True) if h]


def clearance(fp: FloorPlan, p: np.ndarray, q: np.ndarray) -> float:
    return float(np.min(_segment_distance(np.asarray(p, float), np.asarray(q, float), fp.blocking_segs)))


def point_clearance(fp: FloorPlan, p: np.ndarray) -> float:
    return float(np.min(_point_segment_distance(np.asarray(p, float), fp.blocking_segs[:, 0], fp.blocking_segs[:, 1])))


def inside_obstacle(fp: FloorPlan, p: np.ndarray) -> bool:
    return any(_point_in_poly(p, poly) for poly in fp.obstacle_polys)


def room_of(fp: FloorPlan, p: np.ndarray) -> str | None:
    return next((name for name, poly in fp.rooms.items() if _point_in_poly(p, poly)), None)


def is_free_point(fp: FloorPlan, p: np.ndarray, margin: float = BODY_RADIUS_M) -> bool:
    xmin, ymin, xmax, ymax = fp.bounds
    if not (xmin < p[0] < xmax and ymin < p[1] < ymax):
        return False
    return not inside_obstacle(fp, p) and point_clearance(fp, p) >= margin


def sample_free_point(fp: FloorPlan, rng: np.random.Generator, room: str | None = None) -> np.ndarray:
    xmin, ymin, xmax, ymax = fp.bounds
    for _ in range(1000):
        p = rng.uniform((xmin, ymin), (xmax, ymax))
        if is_free_point(fp, p) and (room is None or room_of(fp, p) == room):
            return p
    raise ValueError(f"no free space found in floor plan {fp.name!r} (room={room!r})")


def plan_route(fp: FloorPlan, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
    """Shortest wall-free path as waypoints after `start`, ending at `goal`; None if unreachable."""
    nodes = np.vstack([start, goal, fp.nav_nodes])
    dist = {0: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, 0)]
    done: set[int] = set()
    while heap:
        d, i = heapq.heappop(heap)
        if i in done:
            continue
        if i == 1:
            path = [1]
            while path[-1] != 0:
                path.append(prev[path[-1]])
            return [nodes[k] for k in reversed(path[:-1])]
        done.add(i)
        for j in range(1, len(nodes)):
            if j in done:
                continue
            nd = d + float(np.linalg.norm(nodes[j] - nodes[i]))
            if nd < dist.get(j, np.inf) and clearance(fp, nodes[i], nodes[j]) >= BODY_RADIUS_M:
                dist[j], prev[j] = nd, i
                heapq.heappush(heap, (nd, j))
    return None


def _parse_poly(points: list, name: str) -> np.ndarray:
    poly = np.array(points, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        raise ValueError(f"polygon {name!r} needs at least 3 [x, y] points")
    return poly


def _poly_edges(poly: np.ndarray) -> np.ndarray:
    return np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)


def _door_nodes(doors: np.ndarray) -> np.ndarray:
    mid = doors.mean(axis=1)
    d = doors[:, 1] - doors[:, 0]
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.linalg.norm(d, axis=1, keepdims=True)
    return np.concatenate([mid + DOOR_APPROACH_M * normal, mid - DOOR_APPROACH_M * normal])


def _corner_nodes(poly: np.ndarray) -> np.ndarray:
    # Offset each corner along the sum of its two outward edge normals, so the node keeps
    # NODE_OFFSET_M from both edges of a right-angled corner.
    x, y = poly[:, 0], poly[:, 1]
    ccw = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0
    e = np.roll(poly, -1, axis=0) - poly
    n = np.stack([e[:, 1], -e[:, 0]], axis=1) / np.linalg.norm(e, axis=1, keepdims=True)
    if not ccw:
        n = -n
    return poly + NODE_OFFSET_M * (n + np.roll(n, 1, axis=0))


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def _segments_intersect(p: np.ndarray, q: np.ndarray, segs: np.ndarray) -> np.ndarray:
    a, b = segs[:, 0], segs[:, 1]
    d1, d2 = _cross(a, b, p), _cross(a, b, q)
    d3, d4 = _cross(p, q, a), _cross(p, q, b)
    proper = (d1 * d2 < 0) & (d3 * d4 < 0)
    touching = (
        (np.abs(d1) < EPS) & _on_segment(p, a, b)
        | (np.abs(d2) < EPS) & _on_segment(q, a, b)
        | (np.abs(d3) < EPS) & _on_segment(a, p, q)
        | (np.abs(d4) < EPS) & _on_segment(b, p, q)
    )
    return proper | touching


def _on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo, hi = np.minimum(a, b) - EPS, np.maximum(a, b) + EPS
    return np.all((p >= lo) & (p <= hi), axis=-1)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), EPS), 0.0, 1.0)
    return np.linalg.norm(a + t[..., None] * ab - p, axis=-1)


def _segment_distance(p: np.ndarray, q: np.ndarray, segs: np.ndarray) -> np.ndarray:
    a, b = segs[:, 0], segs[:, 1]
    d = np.minimum.reduce(
        [
            _point_segment_distance(p, a, b),
            _point_segment_distance(q, a, b),
            _point_segment_distance(a, p, q),
            _point_segment_distance(b, p, q),
        ]
    )
    return np.where(_segments_intersect(p, q, segs), 0.0, d)


def _point_in_poly(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = p
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj == yi, EPS, yj - yi) + xi)
    return bool(np.count_nonzero(crosses) % 2)
