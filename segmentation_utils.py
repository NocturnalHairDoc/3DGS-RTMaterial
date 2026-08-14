"""CPU-safe helpers shared by interactive and SAM-driven segmentation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def has_selected_points(mask: torch.Tensor) -> bool:
    """Return ``True`` only when a boolean-like selection contains a point."""
    return bool(torch.count_nonzero(mask).item())


def nearest_visible_points(
    pixel_indices: torch.Tensor,
    depths: torch.Tensor,
    pixel_count: int,
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-4,
) -> torch.Tensor:
    """Approximate center visibility with a per-pixel z-buffer.

    Several Gaussian centers may project to one pixel.  The nearest center and
    centers within a small relative tolerance are retained; clearly occluded
    centers cannot vote for a foreground SAM mask.
    """
    if pixel_indices.numel() == 0:
        return torch.zeros_like(pixel_indices, dtype=torch.bool)
    nearest = torch.full(
        (pixel_count,), torch.inf, dtype=depths.dtype, device=depths.device
    )
    nearest.scatter_reduce_(0, pixel_indices, depths, reduce="amin", include_self=True)
    front = nearest[pixel_indices]
    tolerance = torch.maximum(
        front.abs() * float(relative_tolerance),
        torch.full_like(front, float(absolute_tolerance)),
    )
    return depths <= front + tolerance


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def associate_view_labels(
    view_labels: list[np.ndarray],
    node_sizes: list[int],
    node_views: list[int],
    window: int = 3,
    min_intersection: int = 3,
    iou_threshold: float = 0.15,
    containment_threshold: float = 0.50,
) -> list[int]:
    """Associate view-local mask nodes using their shared visible 3D points.

    ``view_labels[v][p]`` is a globally unique node id or ``-1``.  Only nearby
    views are compared, avoiding quadratic comparisons over every SAM mask.
    """
    dsu = DisjointSet(len(node_sizes))
    for right_view, right in enumerate(view_labels):
        for left_view in range(max(0, right_view - window), right_view):
            left = view_labels[left_view]
            valid = (left >= 0) & (right >= 0)
            if not np.any(valid):
                continue
            pairs, counts = np.unique(
                np.stack((left[valid], right[valid]), axis=1), axis=0,
                return_counts=True,
            )
            for (left_node, right_node), intersection in zip(pairs, counts):
                left_node, right_node = int(left_node), int(right_node)
                intersection = int(intersection)
                if intersection < min_intersection:
                    continue
                union = node_sizes[left_node] + node_sizes[right_node] - intersection
                iou = intersection / max(union, 1)
                containment = intersection / max(min(node_sizes[left_node], node_sizes[right_node]), 1)
                if iou >= iou_threshold or containment >= containment_threshold:
                    dsu.union(left_node, right_node)
    return [dsu.find(node) for node in range(len(node_sizes))]


def component_point_assignments(
    view_labels: list[np.ndarray],
    roots: list[int],
    min_votes: int,
    min_points: int = 10,
) -> list[np.ndarray]:
    """Create disjoint point selections from cross-view components.

    A point contributes at most one vote per view.  If competing components
    contain a point, the component with the most view votes wins.
    """
    if not view_labels:
        return []
    point_votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for labels in view_labels:
        for point_id in np.flatnonzero(labels >= 0):
            point_votes[int(point_id)][roots[int(labels[point_id])]] += 1

    winners: dict[int, list[int]] = defaultdict(list)
    for point_id, votes in point_votes.items():
        root, count = max(votes.items(), key=lambda item: (item[1], -item[0]))
        if count >= min_votes:
            winners[root].append(point_id)
    return [
        np.asarray(points, dtype=np.int64)
        for _, points in sorted(winners.items())
        if len(points) >= min_points
    ]
