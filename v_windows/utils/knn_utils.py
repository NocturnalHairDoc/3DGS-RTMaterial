"""Small PyTorch3D-compatible KNN subset with scalable Windows fallbacks."""

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import torch

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from pytorch3d.ops import knn_points as _pytorch3d_knn_points
except ImportError:
    _pytorch3d_knn_points = None


KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def _ckdtree_knn(p1, p2, K, query_chunk_size=250_000):
    """Exact KNN through a CPU spatial tree, returned on the source device."""
    batch_size, query_count, _ = p1.shape
    device = p1.device
    dists_out = torch.empty((batch_size, query_count, K), dtype=torch.float32, device=device)
    idx_out = torch.empty((batch_size, query_count, K), dtype=torch.long, device=device)

    for batch_id in range(batch_size):
        reference_cpu = p2[batch_id].detach().float().cpu().numpy()
        same_points = p1.shape[1] == p2.shape[1] and p1[batch_id].data_ptr() == p2[batch_id].data_ptr()
        query_cpu = reference_cpu if same_points else p1[batch_id].detach().float().cpu().numpy()

        print(
            f"Building CPU cKDTree for {reference_cpu.shape[0]:,} reference points "
            f"and {query_cpu.shape[0]:,} queries (K={K})...",
            flush=True,
        )
        # Median-balanced/compacted construction is prohibitively slow for
        # multi-million point 3DGS clouds. Sliding-midpoint nodes build much
        # faster while preserving exact cKDTree query results.
        tree = cKDTree(reference_cpu, balanced_tree=False, compact_nodes=False)
        ranges = []
        for start in range(0, query_cpu.shape[0], query_chunk_size):
            stop = min(start + query_chunk_size, query_cpu.shape[0])
            ranges.append((start, stop))

        def query_chunk(bounds):
            start, stop = bounds
            distances, indices = tree.query(query_cpu[start:stop], k=K, workers=1)
            return start, stop, distances, indices

        worker_count = min(16, len(ranges), os.cpu_count() or 1)
        completed = 0
        processed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(query_chunk, bounds) for bounds in ranges]
            for future in as_completed(futures):
                start, stop, distances, indices = future.result()
                completed += 1
                processed += stop - start
                if K == 1:
                    distances = distances[:, None]
                    indices = indices[:, None]
                dists_out[batch_id, start:stop].copy_(
                    torch.from_numpy(distances).to(device=device, dtype=torch.float32).square()
                )
                idx_out[batch_id, start:stop].copy_(
                    torch.from_numpy(indices).to(device=device, dtype=torch.long)
                )
                if completed == len(ranges) or completed % max(1, len(ranges) // 10) == 0:
                    print(f"cKDTree query progress: {processed:,}/{query_cpu.shape[0]:,}", flush=True)

    return dists_out, idx_out


def knn_points(p1, p2, K=1, return_nn=False, chunk_size=4096, **kwargs):
    """Match the ``pytorch3d.ops.knn_points`` subset used by this project."""
    if _pytorch3d_knn_points is not None:
        return _pytorch3d_knn_points(p1, p2, K=K, return_nn=return_nn, **kwargs)
    if p1.ndim != 3 or p2.ndim != 3 or p1.shape[0] != p2.shape[0]:
        raise ValueError("p1 and p2 must have shape (B, N, D) with equal batches")
    if K < 1 or K > p2.shape[1]:
        raise ValueError(f"K must be between 1 and {p2.shape[1]}, got {K}")

    pair_count = p1.shape[0] * p1.shape[1] * p2.shape[1]
    if pair_count > 100_000_000:
        if cKDTree is None:
            raise RuntimeError(
                "Large-point-cloud KNN requires scipy.spatial.cKDTree. "
                "Install scipy or train with --smooth_K 1."
            )
        dists, idx = _ckdtree_knn(p1, p2, K)
    else:
        # Keep each temporary distance matrix below roughly 128 MiB.
        safe_chunk_size = max(1, min(chunk_size, 32_000_000 // p2.shape[1]))
        all_dists, all_idx = [], []
        for start in range(0, p1.shape[1], safe_chunk_size):
            query = p1[:, start:start + safe_chunk_size]
            distances = torch.cdist(query, p2, p=2).square()
            chunk_dists, chunk_idx = torch.topk(distances, K, dim=-1, largest=False, sorted=True)
            all_dists.append(chunk_dists)
            all_idx.append(chunk_idx)
        dists = torch.cat(all_dists, dim=1)
        idx = torch.cat(all_idx, dim=1)
    neighbors = None
    if return_nn:
        batch = torch.arange(p2.shape[0], device=p2.device)[:, None, None]
        neighbors = p2[batch, idx]
    return KNN(dists=dists, idx=idx, knn=neighbors)
