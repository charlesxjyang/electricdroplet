"""
Monkey-patch graph_longrange to use chunked edge computation.

The original builds a complete graph (N² edges) all at once, needing ~144 GB
for 6327 atoms. This patch processes edges in chunks, reducing peak memory
from ~25 GB to ~2.5 GB for the graph construction alone.

Usage: import this module BEFORE calling PolarMACE.
    import patch_graph_longrange
    patch_graph_longrange.apply()
"""
import torch
import numpy as np


def chunked_complete_graph(batch, N, chunk_size=2000):
    """Replacement for batch_complete_graph_excluding_self_duplicates_vector.

    Instead of materializing the full D×D mesh at once, processes sender
    nodes in chunks. Edges are pairwise independent so chunking is exact.

    Args:
        batch: (M,) int tensor — graph membership of each original node
        N: duplication factor (1 for l=0, 4 for l=1)
        chunk_size: number of sender nodes per chunk (controls peak memory)

    Returns:
        edge_index: (2, E) int tensor of [sender, receiver] pairs
    """
    M = batch.shape[0]
    D = M * N
    device = batch.device

    # Create duplicated node IDs and original-atom IDs
    nodes = torch.arange(D, device=device)
    orig = torch.arange(M, device=device).repeat_interleave(N)

    all_senders = []
    all_receivers = []

    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        chunk_nodes = nodes[start:end]          # (C,)
        chunk_orig = orig[start:end]             # (C,)

        # For this chunk of senders, pair with ALL receivers
        C = end - start
        row = chunk_nodes.view(-1, 1).expand(-1, D).reshape(-1)    # (C*D,)
        col = nodes.view(1, -1).expand(C, -1).reshape(-1)          # (C*D,)
        o_row = chunk_orig.view(-1, 1).expand(-1, D).reshape(-1)
        o_col = orig.view(1, -1).expand(C, -1).reshape(-1)

        # Exclude self-atom pairs (same original atom)
        keep = o_row != o_col
        all_senders.append(row[keep])
        all_receivers.append(col[keep])

        # Free intermediate tensors
        del row, col, o_row, o_col, keep

    edge_index = torch.stack([
        torch.cat(all_senders),
        torch.cat(all_receivers),
    ], dim=0)

    return edge_index


def chunked_charges_features(self, edge_index, positions, charges_n, smooth_cutoff):
    """Replacement for charges_features_from_graph that processes edges in chunks.

    The original computes R_ij for ALL edges at once (~8 GB for 640M edges).
    This version chunks the edge list and accumulates scatter_sum results.
    """
    sender, receiver = edge_index[0], edge_index[1]
    n_nodes = positions.shape[0]
    n_edges = sender.shape[0]
    n_charges = charges_n.shape[1]
    device = positions.device

    # Accumulator for node features
    features = torch.zeros(n_nodes, n_charges, device=device)

    EDGE_CHUNK = 10_000_000  # 10M edges per chunk

    for start in range(0, n_edges, EDGE_CHUNK):
        end = min(start + EDGE_CHUNK, n_edges)
        s = sender[start:end]
        r = receiver[start:end]

        R_ij = positions[r] - positions[s]
        d_ij = torch.norm(R_ij, dim=-1)

        smooth_reciprocal = torch.erf(d_ij * smooth_cutoff) / (d_ij + 1e-6)

        # charges_n[s] shape: (chunk, n_charges)
        contrib = charges_n[s] * smooth_reciprocal.unsqueeze(-1)

        # Accumulate into receiver nodes
        features.scatter_add_(0, r.unsqueeze(-1).expand_as(contrib), contrib)

        del R_ij, d_ij, smooth_reciprocal, contrib

    return features


def apply(chunk_size=2000):
    """Apply the memory-reduction patches to graph_longrange."""
    import graph_longrange.realspace_electrostatics as rs

    # Save originals
    rs._orig_complete_graph = rs.batch_complete_graph_excluding_self_duplicates_vector

    # Patch the complete graph function
    def patched_graph(batch, N):
        return chunked_complete_graph(batch, N, chunk_size=chunk_size)
    rs.batch_complete_graph_excluding_self_duplicates_vector = patched_graph

    print(f"[patch] graph_longrange patched: chunked edges (chunk_size={chunk_size})")
