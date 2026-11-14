"""Sharded parameters (FSDP2 / HSDP, through torch.distributed.tensor.DTensor).

With FSDP2 every parameter is a DTensor: each rank holds a shard (its local tensor) of the global parameter, and ranks
on the same mesh coordinates of the sharded dimensions hold the same shard (the replicas of HSDP). Programs still run on
every rank with the same code, but the ranks' parameters are no longer equal, so OPAL compares each shard only with its
replicas, copies and restores local tensors, and runs analysis.py on every rank (a DTensor reduction is a collective).

FSDP1 (FullyShardedDataParallel) flattens parameters into FlatParameters whose names and shapes are not the model's;
it is refused. Plain tensors that a script shards by hand (tensor parallelism without DTensor) are not detected: the
agent would see them as whole parameters.
"""

from __future__ import annotations

import hashlib

import torch

try:  # torch >= 2.4 exposes the public module; older releases keep it under _tensor
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        DTensor = None


def is_dtensor(t) -> bool:
    return DTensor is not None and isinstance(t, DTensor)


def local(t):
    """The tensor this rank stores: the local shard of a DTensor, the tensor itself otherwise."""
    return t.to_local() if is_dtensor(t) else t


def local_data(t):
    """This rank's storage of t, detached from autograd, for in-place writes (t.data for a plain tensor)."""
    return t.detach().to_local() if is_dtensor(t) else t.data


def any_sharded(named) -> bool:
    return any(is_dtensor(p) for _, p in named)


def refuse_fsdp1(model, named):
    """FSDP1 hides the model's parameters behind FlatParameters (or 1-D views of them with use_orig_params): refuse."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
    except ImportError:  # pragma: no cover
        FSDP1 = None
    if FSDP1 is not None and isinstance(model, torch.nn.Module):
        if any(isinstance(m, FSDP1) for m in model.modules()):
            raise ValueError(
                "opal: FSDP1 (torch.distributed.fsdp.FullyShardedDataParallel) is not supported: its flat parameters "
                "hide the model's names and shapes from the agent. Use FSDP2 (torch.distributed.fsdp.fully_shard), "
                "whose parameters are DTensors"
            )
    for n, p in named:
        if getattr(p, "_is_flat_param", False) or type(p).__name__ == "FlatParameter":
            raise ValueError(f"opal: parameter {n} is an FSDP1 FlatParameter; use FSDP2 (fully_shard) instead")


def _mesh_names(mesh):
    names = getattr(mesh, "mesh_dim_names", None)
    return list(names) if names else [f"dim{i}" for i in range(mesh.ndim)]


def describe(p) -> str:
    """Placement of a DTensor for the agent: 'sharded over dp (Shard(0)) ... local (4, 8) on rank 0'; '' otherwise."""
    if not is_dtensor(p):
        return ""
    mesh = p.device_mesh
    parts = [f"{n}={pl}" for n, pl in zip(_mesh_names(mesh), p.placements)]
    return f"DTensor[{', '.join(parts)}] local {tuple(p.to_local().shape)}"


def _shard_coords(p, rank_coords):
    """This rank's coordinates on the mesh dimensions that split the tensor (Shard or Partial placements)."""
    return tuple(c for c, pl in zip(rank_coords, p.placements) if not pl.is_replicate())


def replica_key(p) -> bytes:
    """Ranks with the same key must hold the same local tensor: same mesh, same coordinates on its splitting
    dimensions. Plain tensors: one key, so every rank must agree (data parallelism, as without sharding)."""
    if not is_dtensor(p):
        return _key("plain")
    mesh = p.device_mesh
    coords = mesh.get_coordinate()
    if coords is None:  # this rank is not in the tensor's mesh: it holds nothing of it
        return _key("absent")
    return _key(repr((tuple(mesh.mesh.flatten().tolist()), tuple(mesh.mesh.shape), _shard_coords(p, coords))))


def _key(text) -> bytes:
    """A 16-byte key (every row of the rank check has the same width)."""
    return hashlib.sha256(text.encode()).digest()[:16]


def has_replicas(named, world) -> bool:
    """True if some rank's tensor has a replica elsewhere: a plain parameter (data parallelism) or a DTensor
    placed Replicate on a mesh dimension of size > 1. With pure sharding there is nothing to compare."""
    for _, p in named:
        if not is_dtensor(p):
            if world > 1:
                return True
            continue
        mesh = p.device_mesh
        for size, pl in zip(mesh.mesh.shape, p.placements):
            if pl.is_replicate() and size > 1:
                return True
    return False


def broadcast_from_replica(p):
    """Give every replica of p's shard the values of the first rank of its replica group (in place, on the device)."""
    import torch.distributed as dist

    if not is_dtensor(p):
        dist.broadcast(p.data, src=0)
        return
    mesh = p.device_mesh
    if mesh.get_coordinate() is None:
        return
    t = local_data(p)  # detached: an in-place write into a parameter's shard
    for dim, pl in enumerate(p.placements):
        if pl.is_replicate() and mesh.mesh.shape[dim] > 1:
            group = mesh.get_group(dim)
            dist.broadcast(t, src=dist.get_global_rank(group, 0), group=group)


class ShardCopy:
    """A host copy of a DTensor's local shard and what rebuilds the DTensor (snapshots of updater state)."""

    __slots__ = ("local", "mesh", "placements", "shape", "stride")

    def __init__(self, t):
        self.local = t.to_local().detach().to("cpu", copy=True)
        self.mesh, self.placements = t.device_mesh, tuple(t.placements)
        self.shape, self.stride = tuple(t.shape), tuple(t.stride())

    def nbytes(self):
        return self.local.numel() * self.local.element_size()

    def rebuild(self, device):
        return DTensor.from_local(
            self.local.to(device, copy=True),
            self.mesh,
            self.placements,
            run_check=False,
            shape=torch.Size(self.shape),
            stride=self.stride,
        )


def copy_into(dst, src, device, dtype=None):
    """Put the values of `src` (a host copy: a tensor or a ShardCopy) into `dst` in place where dtype and shape allow,
    so that FSDP keeps reading the same storage; otherwise replace dst's data. Returns dst."""
    dtype = dtype or dst.dtype
    if isinstance(src, ShardCopy):
        mine = local_data(dst) if is_dtensor(dst) else None
        if mine is not None and mine.dtype is dtype and mine.shape == src.local.shape and dst.device == device:
            mine.copy_(src.local)
        else:
            dst.data = src.rebuild(device).to(dtype)
        return dst
    if dst.dtype is not dtype or dst.device != device or dst.shape != src.shape:
        dst.data = src.to(device, dtype)
    else:
        dst.data.copy_(src)
    return dst
