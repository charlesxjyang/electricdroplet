"""
Monkey-patch graph_longrange to use chunked/multi-GPU edge computation.

The original builds a complete graph (N² edges) all at once, needing ~144 GB
for 6327 atoms. This patch fuses graph construction + feature computation:
edges are generated and consumed in chunks, never materialized globally.

With multiple GPUs, chunks are distributed across devices and results reduced.

Usage:
    import patch_graph_longrange
    patch_graph_longrange.apply()  # auto-detects GPU count
"""
import torch
import numpy as np
from scipy.constants import pi


def _fused_scatter_features_chunked(
    charges,            # (D,) float, requires_grad
    positions,          # (D, 3) float, requires_grad
    orig,               # (D,) long — original atom ID per duplicated node
    total_width_factors,  # (1, n_radial) float
    field_constant,     # scalar
    chunk_size=2000,    # sender nodes per chunk
    device=None,
):
    """Compute scatter_sum features without materializing full edge_index.

    For each chunk of sender nodes, generates sender-receiver pairs on the fly,
    computes edge features, scatter-sums into the output, then frees the chunk.

    This is mathematically identical to:
        edge_index = batch_complete_graph(batch, N)
        features = charges_features_from_graph(charges, positions, edge_index, ...)
    but uses O(chunk_size * D) memory instead of O(D²).
    """
    D = positions.shape[0]
    n_radial = total_width_factors.shape[1]
    if device is None:
        device = positions.device

    features = torch.zeros(D, n_radial, dtype=positions.dtype, device=device)
    all_nodes = torch.arange(D, device=device)

    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        C = end - start

        # Sender indices for this chunk
        sender_nodes = all_nodes[start:end]       # (C,)
        sender_orig = orig[start:end]              # (C,)

        # Pair each sender with ALL receivers — build on the fly
        # s: (C*D,), r: (C*D,)
        s = sender_nodes.unsqueeze(1).expand(C, D).reshape(-1)
        r = all_nodes.unsqueeze(0).expand(C, D).reshape(-1)

        # Exclude self-atom pairs (same original atom)
        s_orig = sender_orig.unsqueeze(1).expand(C, D).reshape(-1)
        r_orig = orig.unsqueeze(0).expand(C, D).reshape(-1)
        keep = s_orig != r_orig
        s = s[keep]
        r = r[keep]
        del s_orig, r_orig, keep

        # Compute edge features (WITH gradient tracking for forces)
        R_ij = positions[s] - positions[r]        # (E_chunk, 3)
        d_ij = torch.norm(R_ij, dim=-1, keepdim=True)  # (E_chunk, 1)
        smooth_recip = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)

        # Scatter-sum contribution from this chunk
        contrib = charges[s].unsqueeze(-1) * smooth_recip  # (E_chunk, n_radial)
        features = features + torch.zeros_like(features).scatter_add_(
            0, r.unsqueeze(-1).expand_as(contrib), contrib
        )

        del R_ij, d_ij, smooth_recip, contrib, s, r

    features = field_constant * features / (4 * pi)
    return features


def _fused_scatter_features_multigpu(
    charges, positions, orig, total_width_factors, field_constant,
    chunk_size=2000, n_gpus=None,
):
    """Multi-GPU version: partition sender nodes across GPUs.

    Each GPU computes partial scatter_sum for its assigned senders,
    then results are reduced (summed) on GPU 0.

    Gradient flow: positions and charges on GPU 0 have .to(device_k)
    copies that maintain autograd connectivity. The final .to(device_0)
    and sum preserves the full gradient chain for force computation.
    """
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return _fused_scatter_features_chunked(
            charges, positions, orig, total_width_factors, field_constant,
            chunk_size=chunk_size,
        )

    D = positions.shape[0]
    n_radial = total_width_factors.shape[1]
    device_0 = positions.device

    # Partition sender nodes across GPUs
    senders_per_gpu = (D + n_gpus - 1) // n_gpus
    partials = []

    for gpu_id in range(n_gpus):
        start = gpu_id * senders_per_gpu
        end = min(start + senders_per_gpu, D)
        if start >= D:
            break

        target = torch.device(f'cuda:{gpu_id}')

        # Replicate tensors to this GPU (maintains autograd for .to())
        pos_k = positions.to(target)
        chg_k = charges.to(target)
        orig_k = orig.to(target)
        twf_k = total_width_factors.to(target)

        # Compute partial features for this GPU's sender range
        partial = _fused_scatter_features_for_range(
            chg_k, pos_k, orig_k, twf_k, field_constant,
            start, end, chunk_size,
        )
        partials.append(partial)

    # Reduce to GPU 0
    result = partials[0].to(device_0)
    for p in partials[1:]:
        result = result + p.to(device_0)

    return result


def _fused_scatter_features_for_range(
    charges, positions, orig, total_width_factors, field_constant,
    sender_start, sender_end, chunk_size,
):
    """Compute features for a specific range of sender nodes on one GPU."""
    D = positions.shape[0]
    n_radial = total_width_factors.shape[1]
    device = positions.device

    features = torch.zeros(D, n_radial, dtype=positions.dtype, device=device)
    all_nodes = torch.arange(D, device=device)

    for start in range(sender_start, sender_end, chunk_size):
        end = min(start + chunk_size, sender_end)
        C = end - start

        sender_nodes = all_nodes[start:end]
        sender_orig = orig[start:end]

        s = sender_nodes.unsqueeze(1).expand(C, D).reshape(-1)
        r = all_nodes.unsqueeze(0).expand(C, D).reshape(-1)

        s_orig = sender_orig.unsqueeze(1).expand(C, D).reshape(-1)
        r_orig = orig.unsqueeze(0).expand(C, D).reshape(-1)
        keep = s_orig != r_orig
        s = s[keep]
        r = r[keep]
        del s_orig, r_orig, keep

        R_ij = positions[s] - positions[r]
        d_ij = torch.norm(R_ij, dim=-1, keepdim=True)
        smooth_recip = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)

        contrib = charges[s].unsqueeze(-1) * smooth_recip
        features = features + torch.zeros_like(features).scatter_add_(
            0, r.unsqueeze(-1).expand_as(contrib), contrib
        )

        del R_ij, d_ij, smooth_recip, contrib, s, r

    features = field_constant * features / (4 * pi)
    return features


def _patched_call_density_0_feats_0(self, source_feats, positions, batch):
    """Replacement for call_density_0_feats_0 — fused, no full edge_index."""
    M = batch.shape[0]
    orig = torch.arange(M, device=batch.device)

    features = _DISPATCH_FN(
        charges=source_feats[:, 0],
        positions=positions,
        orig=orig,
        total_width_factors=self.total_width_factors.unsqueeze(0),
        field_constant=_FIELD_CONSTANT,
        chunk_size=_CHUNK_SIZE,
    )
    return self.l0_factors * features


def _patched_call_density_1_feats_1(self, source_feats, positions, batch):
    """Replacement for call_density_1_feats_1 — fused, no full edge_index."""
    extended_positions = positions.repeat_interleave(4, dim=0)
    extended_positions[1::4] = extended_positions[1::4] + self.x
    extended_positions[2::4] = extended_positions[2::4] + self.y
    extended_positions[3::4] = extended_positions[3::4] + self.z

    charges = torch.zeros_like(extended_positions[:, 0])
    charges[1::4] = source_feats[:, 3] / self.offset
    charges[2::4] = source_feats[:, 1] / self.offset
    charges[3::4] = source_feats[:, 2] / self.offset
    charges[0::4] = source_feats[:, 0] - (
        charges[1::4] + charges[2::4] + charges[3::4]
    )

    M = batch.shape[0]
    orig = torch.arange(M, device=batch.device).repeat_interleave(4)

    scalar_features = _DISPATCH_FN(
        charges=charges,
        positions=extended_positions,
        orig=orig,
        total_width_factors=self.total_width_factors.unsqueeze(0),
        field_constant=_FIELD_CONSTANT,
        chunk_size=_CHUNK_SIZE,
    )

    all_features = torch.zeros(
        batch.size(0),
        4 * self.num_radial,
        dtype=torch.get_default_dtype(),
        device=batch.device,
    )
    all_features[:, :self.num_radial] = self.l0_factors * scalar_features[0::4]
    all_features[:, self.num_radial::3] = self.l1_factors * (
        scalar_features[2::4] - scalar_features[0::4]
    )
    all_features[:, self.num_radial + 1::3] = self.l1_factors * (
        scalar_features[3::4] - scalar_features[0::4]
    )
    all_features[:, self.num_radial + 2::3] = self.l1_factors * (
        scalar_features[1::4] - scalar_features[0::4]
    )
    return all_features


# Module-level config set by apply()
_DISPATCH_FN = None
_FIELD_CONSTANT = None
_CHUNK_SIZE = 2000


def apply(chunk_size=2000, force_single_gpu=False):
    """Apply memory-reduction patches to graph_longrange.

    Auto-detects GPU count. With >1 GPU, uses multi-GPU dispatch.
    With 1 GPU, uses chunked single-GPU (still saves memory vs original).
    """
    global _DISPATCH_FN, _FIELD_CONSTANT, _CHUNK_SIZE

    import graph_longrange.realspace_electrostatics as rs
    from graph_longrange.utils import FIELD_CONSTANT

    _FIELD_CONSTANT = FIELD_CONSTANT
    _CHUNK_SIZE = chunk_size

    n_gpus = torch.cuda.device_count()
    use_multi = (n_gpus > 1) and (not force_single_gpu)

    if use_multi:
        _DISPATCH_FN = lambda **kw: _fused_scatter_features_multigpu(
            n_gpus=n_gpus, **kw
        )
        mode = "multi-GPU ({} GPUs)".format(n_gpus)
    else:
        _DISPATCH_FN = _fused_scatter_features_chunked
        mode = "chunked single-GPU"

    # Patch the class methods
    rs.RealSpaceFiniteDifferenceElectrostaticFeatures.call_density_0_feats_0 = \
        _patched_call_density_0_feats_0
    rs.RealSpaceFiniteDifferenceElectrostaticFeatures.call_density_1_feats_1 = \
        _patched_call_density_1_feats_1

    print("[patch] graph_longrange: {} (chunk_size={})".format(mode, chunk_size))
