"""GPU sharded-parameter check (FSDP2 fully_shard, NCCL, bf16 mixed precision), run with torchrun on 2+ GPUs:

    torchrun --nproc_per_node=8 [--nnodes=2 ...] tests/gpu_fsdp.py --mode plain|keep|swap|agent \
        [--mesh shard|hsdp] [--steps 400] [--deterministic] [--out result.json]

plain: no wrapper. keep: wrapper with the starting program, every decision keeps it. local: a fake agent installs an updater written as the prompt recommends for sharded runs (_foreach_* on local
shards, zeros_like state, a global norm through full_tensor()). swap: a fake agent replaces the
updater at its first decision with a hand-written AdamW equivalent to the starting one (blocking, so the swap lands at
the same update every time): with --deterministic its losses and parameters must match plain bit for bit. agent: a fake
agent rewrites updater.py twice with other learning rates, without blocking training. hsdp: replicate across nodes,
shard within a node (the rank check then compares replicas). Prints median steps/s and a JSON summary from rank 0.
"""

import argparse
import hashlib
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

EQUIV_ADAMW = """
import copy

import torch


class Updater:
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.user = spec["user_optimizers"][0]
        g = self.user.param_groups[0]
        self.opt = torch.optim.AdamW([p for _, p in named_params], lr=g["lr"], betas=g["betas"], eps=g["eps"],
                                     weight_decay=g["weight_decay"], foreach=g["foreach"])
        self.opt.load_state_dict(copy.deepcopy(self.user.state_dict()))
        self.scale = SCALE

    @property
    def param_groups(self):
        return self.opt.param_groups

    def step(self, grads, update):
        for g in self.opt.param_groups:
            g["lr"] = self.user.param_groups[0]["lr"] * self.scale
        self.opt.step()

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, s):
        self.opt.load_state_dict(s)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""

# written the way the agent prompt's sharded section recommends (same as tests/test_sharded.py)
LOCAL_RULE = """
import torch


class Updater:
    follow_user_scheduler = False

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.m = [torch.zeros_like(p) for _, p in named_params]
        self.v = [torch.zeros_like(p) for _, p in named_params]

    @property
    def param_groups(self):
        return [dict(params=[p for _, p in self.named], lr=self.lr(1))]

    @staticmethod
    def lr(update):
        return 5e-3 * min(1.0, update / 10)

    def step(self, grads, update):
        gs = [grads[n] for n, _ in self.named]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs))).full_tensor()
        scale = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
        local = lambda xs: [x.to_local() for x in xs]
        p, g, m, v = local([q for _, q in self.named]), local(gs), local(self.m), local(self.v)
        g = torch._foreach_mul(g, scale)
        torch._foreach_lerp_(m, g, 0.1)
        torch._foreach_mul_(v, 0.99)
        torch._foreach_addcmul_(v, g, g, value=0.01)
        denom = torch._foreach_add(torch._foreach_sqrt(v), 1e-8)
        torch._foreach_addcdiv_(p, m, denom, value=-self.lr(update))

    def state_dict(self):
        return dict(m=self.m, v=self.v)

    def load_state_dict(self, state):
        for mine, saved in zip(self.m + self.v, state["m"] + state["v"]):
            mine.copy_(saved)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        y = self.ln1(x)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab=4096, d=384, layers=6, heads=6, ctx=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, ctx, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, : idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def shard_digest(model, rank, world):
    """SHA-256 over every rank's local shards (in rank order), and whether replicas of the same shard agree."""
    import opal.misc.sharding as sh

    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().to_local().float().cpu().numpy().tobytes())
    mine = (h.hexdigest(), "".join(sh.replica_key(p).hex() for p in model.parameters()))
    every = [None] * world
    dist.all_gather_object(every, mine)
    by_key = {}
    agree = all(by_key.setdefault(key, d) == d for d, key in every)
    total = hashlib.sha256("".join(d for d, _ in every).encode()).hexdigest()[:16]
    return total, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "keep", "swap", "agent", "local"], required=True)
    ap.add_argument("--mesh", choices=["shard", "hsdp"], default="shard")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--agent_delay", type=float, default=2.0, help="seconds the fake agent takes per decision")
    ap.add_argument("--deterministic", action="store_true", help="deterministic kernels, for bitwise checks")
    ap.add_argument("--save_at", type=int, default=None, help="save a per-rank checkpoint after this step")
    ap.add_argument("--resume", default=None, help="folder of a --save_at checkpoint to resume from")
    a = ap.parse_args()

    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if a.mesh == "hsdp":
        mesh = init_device_mesh("cuda", (world // per_node, per_node), mesh_dim_names=("rep", "dp"))
    else:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    torch.manual_seed(0)
    model = TinyLM().to(dev)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in model.blocks:
        fully_shard(b, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    lr = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))

    wrap_s = None
    if a.mode != "plain":
        import opal

        def agent(ws, msg, system):
            time.sleep(a.agent_delay)
            if a.mode == "swap" and "Decision 1." in msg:
                (ws / "updater.py").write_text(EQUIV_ADAMW.replace("SCALE", "1.0"))
            if a.mode == "local" and "Decision 1." in msg:
                (ws / "updater.py").write_text(LOCAL_RULE)
            if a.mode == "agent" and ("Decision 1." in msg or "Decision 3." in msg):
                (ws / "updater.py").write_text(EQUIV_ADAMW.replace("SCALE", "0.5" if "Decision 1." in msg else "0.25"))
            return "fake agent"

        backend = "keep" if a.mode == "keep" else agent
        t0 = time.perf_counter()
        opt = opal.wrap(
            opt,
            model,
            scheduler=sched,
            total_steps=a.steps,
            goal="GPU FSDP2 test",
            backend=backend,
            decisions=6,
            every=25,
            sync_every=20,
            rank_check_every=100,
            blocking=a.mode != "agent",
            summarize=False,
            run_dir=a.run_dir,
        )
        wrap_s = round(time.perf_counter() - t0, 2)

    g = torch.Generator(device=dev).manual_seed(1000 + rank)
    times, losses, start = [], [], 0
    if a.resume:
        ck = torch.load(os.path.join(a.resume, f"ck-r{rank}.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if a.mode == "plain":
            sched.load_state_dict(ck["sched"])
        g.set_state(ck["gen"].cpu())
        start, losses = ck["step"], [t.to(dev) for t in ck["losses"]]
    torch.cuda.synchronize()
    t_start = time.time()
    for step in range(start, a.steps):
        t0 = time.perf_counter()
        idx = torch.randint(0, 4096, (a.batch, 257), device=dev, generator=g)
        logits = model(idx[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
        loss.backward()
        opt.step(loss=loss) if a.mode != "plain" else opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(loss.detach())
        if step % 50 == 49:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if a.save_at is not None and step + 1 == a.save_at:
            out = os.path.dirname(a.out) if a.out else "."
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=g.get_state(),
                            step=step + 1, losses=[t.cpu() for t in losses]), os.path.join(out, f"ck-r{rank}.pt"))
            break
    torch.cuda.synchronize()
    wall = time.time() - t_start
    params_sha, replicas_agree = shard_digest(model, rank, world)
    steady = sorted(times[100:]) if len(times) > 200 else sorted(times)
    res = dict(
        mode=a.mode,
        mesh=a.mesh,
        world=world,
        steps=a.steps,
        wrap_s=wrap_s,
        wall_s=round(wall, 2),
        median_steps_per_s=round(1 / statistics.median(steady), 2),
        replicas_agree=replicas_agree,
        params_sha=params_sha,
        loss_sha=hashlib.sha256(torch.stack(losses).float().cpu().numpy().tobytes()).hexdigest()[:16],
        final_loss=float(torch.stack(losses[-20:]).mean()),
        peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        version=getattr(opt, "version", None),
        errors=[e.get("error", "")[:300] for e in getattr(opt, "errors", [])],
        gpu=torch.cuda.get_device_name(dev),
    )
    if rank == 0:
        print(json.dumps(res))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
