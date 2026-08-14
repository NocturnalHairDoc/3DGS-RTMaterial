"""Lightweight cross-view instance graph for SAM/SAGA Gaussian segmentation.

The graph deliberately operates on spatial anchors instead of individual
Gaussians.  Gaussian-level decisions are made only for anchors on an instance
boundary, keeping the expensive part small and auditable.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from segmentation_utils import DisjointSet


@dataclass
class AnchorGraph:
    point_to_anchor: np.ndarray
    centroids: np.ndarray
    features: np.ndarray
    sizes: np.ndarray
    edges: np.ndarray
    edge_weights: np.ndarray
    voxel_size: float


@dataclass
class MaskNode:
    view: int
    local_mask: int
    anchors: np.ndarray
    support: np.ndarray
    context: np.ndarray
    saga: np.ndarray
    points: np.ndarray


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _voxel_partition(xyz: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    origin = xyz.min(axis=0)
    keys = np.floor((xyz - origin) / max(voxel_size, 1e-8)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.astype(np.int64), origin


def build_anchor_graph(
    xyz: np.ndarray,
    features: np.ndarray,
    target_anchors: int = 1600,
    neighbours: int = 8,
    max_anchor_points: int = 128,
) -> AnchorGraph:
    """Voxelise Gaussians into compact anchors and build a sparse spatial graph."""
    xyz = np.asarray(xyz, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("xyz must have shape (N, 3) with N > 0")
    if features.ndim != 2 or features.shape[0] != xyz.shape[0]:
        raise ValueError("features must have shape (N, D) and match xyz")

    target = int(np.clip(target_anchors, 1, xyz.shape[0]))
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    diagonal = max(float(np.linalg.norm(extent)), 1e-6)
    low, high = diagonal * 1e-7, diagonal * 2.0
    best = None
    # Binary search works for volumetric, surface-like, and nearly linear
    # geometry; a volume-derived initial estimate does not.
    for _ in range(28):
        size = (low + high) * 0.5
        candidate, _ = _voxel_partition(xyz, size)
        count = int(candidate.max()) + 1
        error = abs(count - target)
        if best is None or error < best[0]:
            best = (error, size, candidate)
        if count > target:
            low = size
        elif count < target:
            high = size
        else:
            best = (0, size, candidate)
            break
    _, size, inverse = best
    # A few dense voxels can otherwise contain thousands of Gaussians. Split
    # them along their longest local axis so every superpoint remains small.
    if max_anchor_points > 0:
        refined = np.empty_like(inverse)
        next_anchor = 0
        for anchor in range(int(inverse.max()) + 1):
            point_ids = np.flatnonzero(inverse == anchor)
            chunk_count = int(np.ceil(point_ids.size / max_anchor_points))
            if chunk_count > 1:
                local = xyz[point_ids]
                axis = int(np.argmax(np.ptp(local, axis=0)))
                point_ids = point_ids[np.argsort(local[:, axis], kind="stable")]
            for chunk in np.array_split(point_ids, chunk_count):
                refined[chunk] = next_anchor
                next_anchor += 1
        inverse = refined

    anchor_count = int(inverse.max()) + 1
    sizes = np.bincount(inverse, minlength=anchor_count).astype(np.int64)
    centroids = np.zeros((anchor_count, 3), dtype=np.float32)
    anchor_features = np.zeros((anchor_count, features.shape[1]), dtype=np.float32)
    np.add.at(centroids, inverse, xyz)
    np.add.at(anchor_features, inverse, features)
    centroids /= sizes[:, None]
    anchor_features = _normalise_rows(anchor_features / sizes[:, None])

    if anchor_count == 1:
        edges = np.empty((0, 2), dtype=np.int64)
        weights = np.empty(0, dtype=np.float32)
    else:
        k = min(max(int(neighbours), 1) + 1, anchor_count)
        distances, indices = cKDTree(centroids).query(centroids, k=k)
        pairs = set()
        for left in range(anchor_count):
            for right in np.atleast_1d(indices[left])[1:]:
                right = int(right)
                if right != left:
                    pairs.add((min(left, right), max(left, right)))
        edges = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
        delta = centroids[edges[:, 0]] - centroids[edges[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        spatial = np.exp(-0.5 * (distance / max(2.5 * size, 1e-6)) ** 2)
        cosine = np.sum(anchor_features[edges[:, 0]] * anchor_features[edges[:, 1]], axis=1)
        weights = (spatial * (0.5 + 0.5 * np.clip(cosine, -1.0, 1.0))).astype(np.float32)

    return AnchorGraph(inverse, centroids, anchor_features, sizes, edges, weights, size)


@torch.no_grad()
def mask_context_features(image: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
    """Describe mask appearance and its surrounding ring without a heavy encoder."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    masks = masks.to(device=image.device, dtype=torch.bool)
    image = image[:3].float()
    count = masks.flatten(1).sum(1).clamp_min(1).float()
    inside_mean = (masks[:, None] * image[None]).flatten(2).sum(2) / count[:, None]
    centred = image[None] - inside_mean[:, :, None, None]
    inside_std = torch.sqrt(
        (masks[:, None] * centred.square()).flatten(2).sum(2) / count[:, None] + 1e-8
    )
    dilated = F.max_pool2d(masks[:, None].float(), kernel_size=11, stride=1, padding=5)[:, 0] > 0
    ring = dilated & ~masks
    ring_count = ring.flatten(1).sum(1).clamp_min(1).float()
    ring_mean = (ring[:, None] * image[None]).flatten(2).sum(2) / ring_count[:, None]

    h, w = masks.shape[-2:]
    geometry = []
    for mask in masks:
        yy, xx = torch.where(mask)
        if yy.numel():
            geometry.append(torch.stack((
                xx.min() / max(w - 1, 1), yy.min() / max(h - 1, 1),
                xx.max() / max(w - 1, 1), yy.max() / max(h - 1, 1),
                mask.float().mean(),
            )))
        else:
            geometry.append(torch.zeros(5, device=image.device))
    result = torch.cat((inside_mean, inside_std, ring_mean, torch.stack(geometry)), dim=1)
    return _normalise_rows(result.detach().cpu().numpy())


def make_mask_nodes(
    graph: AnchorGraph,
    point_masks: np.ndarray,
    visible_points: np.ndarray,
    contexts: np.ndarray,
    view: int,
    min_anchor_fraction: float = 0.12,
) -> list[MaskNode]:
    """Aggregate one view's point observations into mask-to-anchor votes."""
    visible_counts = np.bincount(
        graph.point_to_anchor[np.asarray(visible_points, dtype=np.int64)],
        minlength=graph.centroids.shape[0],
    )
    nodes = []
    for local_mask in np.unique(point_masks[point_masks >= 0]):
        points = np.flatnonzero(point_masks == local_mask)
        counts = np.bincount(graph.point_to_anchor[points], minlength=len(visible_counts))
        support = counts / np.maximum(visible_counts, 1)
        anchors = np.flatnonzero((counts > 0) & (support >= min_anchor_fraction))
        if anchors.size == 0:
            continue
        weights = support[anchors].astype(np.float32)
        saga = _normalise_rows((graph.features[anchors] * weights[:, None]).sum(0, keepdims=True))[0]
        nodes.append(MaskNode(
            int(view), int(local_mask), anchors.astype(np.int64), weights,
            contexts[int(local_mask)].astype(np.float32), saga,
            points.astype(np.int64),
        ))
    return nodes


def _sparse_overlap(left: MaskNode, right: MaskNode) -> tuple[float, float]:
    common, li, ri = np.intersect1d(left.anchors, right.anchors, return_indices=True)
    if common.size == 0:
        return 0.0, 0.0
    intersection = float(np.minimum(left.support[li], right.support[ri]).sum())
    left_total, right_total = float(left.support.sum()), float(right.support.sum())
    union = left_total + right_total - intersection
    return intersection / max(union, 1e-8), intersection / max(min(left_total, right_total), 1e-8)


def associate_mask_nodes(
    nodes: list[MaskNode],
    graph: AnchorGraph,
    window: int = 3,
    score_threshold: float = 0.50,
) -> tuple[np.ndarray, list[dict]]:
    """Fuse 3D overlap, visibility, context, SAGA and connectivity evidence."""
    dsu = DisjointSet(len(nodes))
    anchor_neighbours = defaultdict(set)
    for left, right in graph.edges:
        anchor_neighbours[int(left)].add(int(right))
        anchor_neighbours[int(right)].add(int(left))
    evidence = []
    for right_id, right in enumerate(nodes):
        for left_id in range(right_id):
            left = nodes[left_id]
            if left.view == right.view or right.view - left.view > window:
                continue
            overlap, visibility = _sparse_overlap(left, right)
            left_set, right_set = set(left.anchors.tolist()), set(right.anchors.tolist())
            connected = 1.0 if left_set & right_set else float(any(
                anchor_neighbours[a] & right_set for a in left_set
            ))
            if overlap == 0.0 and visibility < 0.08 and not connected:
                continue
            context = float(np.clip(np.dot(left.context, right.context), -1, 1) * 0.5 + 0.5)
            saga = float(np.clip(np.dot(left.saga, right.saga), -1, 1) * 0.5 + 0.5)
            score = 0.35 * overlap + 0.15 * visibility + 0.15 * context + 0.20 * saga + 0.15 * connected
            evidence.append({"left": left_id, "right": right_id, "overlap": overlap,
                             "visibility": visibility, "context": context, "saga": saga,
                             "connectivity": connected, "score": score})
            if score >= score_threshold and (overlap >= 0.05 or visibility >= 0.25):
                dsu.union(left_id, right_id)
    return np.asarray([dsu.find(i) for i in range(len(nodes))], dtype=np.int64), evidence


def solve_anchor_instances(
    nodes: list[MaskNode],
    roots: np.ndarray,
    graph: AnchorGraph,
    min_views: int = 2,
    smoothness: float = 0.35,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Use unary multi-view votes plus sparse-graph ICM and connectivity cuts."""
    root_values = sorted(set(roots.tolist()))
    root_to_col = {root: col for col, root in enumerate(root_values)}
    unary = np.zeros((graph.centroids.shape[0], len(root_values)), dtype=np.float32)
    views = defaultdict(set)
    for node, root in zip(nodes, roots):
        col = root_to_col[int(root)]
        unary[node.anchors, col] += node.support
        views[int(root)].add(node.view)
    for root, col in root_to_col.items():
        if len(views[root]) < min_views:
            unary[:, col] = 0

    labels = np.where(unary.max(1) > 0, unary.argmax(1), -1).astype(np.int64)
    neighbours = [[] for _ in range(len(labels))]
    for (left, right), weight in zip(graph.edges, graph.edge_weights):
        if weight <= 0.02:
            continue
        neighbours[int(left)].append((int(right), float(weight)))
        neighbours[int(right)].append((int(left), float(weight)))
    for _ in range(iterations):
        changed = 0
        for anchor in range(len(labels)):
            # Pairwise smoothing may resolve competing observed labels, but it
            # must not flood unobserved/background anchors across the scene.
            if not np.any(unary[anchor] > 0):
                continue
            candidates = set(np.flatnonzero(unary[anchor] > 0).tolist())
            candidates.update(
                labels[n] for n, _ in neighbours[anchor]
                if labels[n] >= 0 and np.any(unary[n] > 0)
            )
            if not candidates:
                continue
            best = max(candidates, key=lambda label: (
                unary[anchor, label] + smoothness * sum(
                    weight for n, weight in neighbours[anchor] if labels[n] == label
                ), -label
            ))
            if labels[anchor] != best:
                labels[anchor] = best
                changed += 1
        if changed == 0:
            break

    # Graph cut: disconnected islands carrying the same semantic component are
    # separate instances. This prevents transitive cross-view chaining.
    instance = np.full_like(labels, -1)
    next_instance = 0
    for label in sorted(set(labels[labels >= 0].tolist())):
        remaining = set(np.flatnonzero(labels == label).tolist())
        while remaining:
            seed = remaining.pop()
            component, queue = [seed], deque([seed])
            while queue:
                current = queue.popleft()
                for other, weight in neighbours[current]:
                    if weight >= 0.05 and other in remaining and labels[other] == label:
                        remaining.remove(other)
                        component.append(other)
                        queue.append(other)
            instance[component] = next_instance
            next_instance += 1
    root_to_instance = {
        root: int(instance[np.argmax(unary[:, col])])
        for root, col in root_to_col.items() if np.any(unary[:, col] > 0)
    }
    return instance, unary, root_to_instance


def refine_boundary_gaussians(
    graph: AnchorGraph,
    anchor_instances: np.ndarray,
    point_features: np.ndarray,
    nodes: list[MaskNode],
    roots: np.ndarray,
    root_to_instance: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Refine only anchors adjacent to a different label, using point SAGA evidence."""
    boundary = np.zeros(len(anchor_instances), dtype=bool)
    for left, right in graph.edges:
        if (anchor_instances[left] >= 0 and anchor_instances[right] >= 0
                and anchor_instances[left] != anchor_instances[right]):
            boundary[left] = boundary[right] = True
    point_labels = anchor_instances[graph.point_to_anchor].copy()
    if not boundary.any():
        return point_labels, boundary

    point_features = _normalise_rows(point_features)
    prototypes = {}
    for instance in np.unique(anchor_instances[anchor_instances >= 0]):
        interior = (anchor_instances == instance) & ~boundary
        selected = np.flatnonzero(interior[graph.point_to_anchor])
        if selected.size == 0:
            selected = np.flatnonzero(anchor_instances[graph.point_to_anchor] == instance)
        prototypes[int(instance)] = _normalise_rows(point_features[selected].mean(0, keepdims=True))[0]

    point_votes = defaultdict(lambda: defaultdict(float))
    for node in nodes:
        node_instances = anchor_instances[node.anchors]
        valid = node_instances >= 0
        if not np.any(valid):
            continue
        candidates = np.unique(node_instances[valid])
        instance = max(candidates, key=lambda value: float(
            node.support[valid & (node_instances == value)].sum()
        ))
        for point in node.points:
            if boundary[graph.point_to_anchor[point]]:
                point_votes[int(point)][int(instance)] += 1.0

    neighbour_instances = defaultdict(set)
    for left, right in graph.edges:
        if anchor_instances[right] >= 0:
            neighbour_instances[int(left)].add(int(anchor_instances[right]))
        if anchor_instances[left] >= 0:
            neighbour_instances[int(right)].add(int(anchor_instances[left]))

    for point in np.flatnonzero(boundary[graph.point_to_anchor]):
        anchor = graph.point_to_anchor[point]
        candidates = set(point_votes[int(point)])
        candidates.update(neighbour_instances[int(anchor)])
        if candidates:
            point_labels[point] = max(candidates, key=lambda instance: (
                point_votes[int(point)].get(instance, 0.0)
                + 0.5 * float(np.dot(point_features[point], prototypes[instance])), -instance
            ))
    return point_labels, boundary
