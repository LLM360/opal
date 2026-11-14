"""Sharded parameters (FSDP2 fully_shard, DTensor) on CPU with gloo: the wrapped run matches the plain run, program
swaps, trials, exact undo, skips, analysis on every rank and checkpoints work on local shards.

Each case trains a wrapped model and a plain reference side by side in the same processes (same mesh, same batches)
and compares their losses bit for bit. Meshes: "shard" (2 ranks, every tensor split in two), "replicate" (2 ranks,
HSDP with replicas only) and "hsdp" (4 ranks: 2 replicas x 2 shards). The last layer's output dimension is 1, so the
second shard of its weight and bias is empty.
"""

import json
import math
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
except ImportError:  # torch without FSDP2's public API
    pytest.skip("FSDP2 (torch.distributed.fsdp.fully_shard) is not available", allow_module_level=True)

import opal  # noqa: E402

TOTAL = 30

EQUIV = """
import copy

import torch


class Updater:
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named, self.rank = named_params, spec["rank"]
        self.user = spec["user_optimizers"][0]
        g = self.user.param_groups[0]
        self.opt = torch.optim.AdamW(
            [p for _, p in named_params], lr=g["lr"], betas=g["betas"], eps=g["eps"], weight_decay=g["weight_decay"]
        )
        self.opt.load_state_dict(copy.deepcopy(self.user.state_dict()))

    @property
    def param_groups(self):
        return self.opt.param_groups

    def step(self, grads, update):
        for g in self.opt.param_groups:
            g["lr"] = self.user.param_groups[0]["lr"]
        self.opt.step()
        EXTRA

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state):
        self.opt.load_state_dict(state)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""
DIVERGE = EQUIV.replace(
    "EXTRA",
    "with torch.no_grad():\n            for _, p in self.named:\n                p.to_local().add_(1e-3 * (self.rank + 1))",
)
FAIL_FIRST = EQUIV.replace("EXTRA", "").replace(
    "        self.opt.step()\n",
    "        if self.rank == 1:\n            raise RuntimeError('fails on rank 1')\n        self.opt.step()\n",
)
EQUIV = EQUIV.replace("EXTRA", "")

SKIP = """
class Updater:
    follow_user_scheduler = True
    can_skip = True

    def __init__(self, named_params, previous=None, spec=None):
        self.opts = spec["user_optimizers"]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, grads, update):
        for o in self.opts:
            o.step()
        return update == SKIP_AT

    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts])

    def load_state_dict(self, state):
        for o, s in zip(self.opts, state["opts"]):
            o.load_state_dict(s)


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
"""

# An updater written the way the prompt's sharded section recommends: elementwise work with torch._foreach_* on local
# shards, state made with zeros_like (sharded like the parameter), one global gradient norm read with full_tensor().
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

EXAMPLE = (Path(opal.__file__).parent / "prompts" / "example_updater.py").read_text()

ANALYSIS = """
import torch
import torch.distributed as dist

EVERY = 5


def analyze(obs, memory):
    report = dict(update=obs.update)
    if obs.losses and max(obs.losses) > -1.0:  # a branch on rank 0's losses, the same on every rank
        gs = [g for g in obs.grads.values() if g is not None]
        total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs)))
        report["grad_norm"] = total.full_tensor().item()
        report["n_losses"] = len(obs.losses)
    RAISE  # after the collectives: raising before one would stall the other ranks (documented)
    return {"report": report}
"""


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = torch.nn.Linear(d, d)
        self.mlp_fc = torch.nn.Linear(d, 2 * d)
        self.mlp_out = torch.nn.Linear(2 * d, d)
        self.norm = torch.nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x + self.mlp_out(torch.relu(self.mlp_fc(x)))


class Net(torch.nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(Block(d) for _ in range(n))
        self.head = torch.nn.Linear(d, 1)  # output dim 1: the second shard is empty

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return self.head(x).squeeze(-1)


def _mesh(kind, world):
    if kind in ("shard", "mixed"):
        return init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
    if kind == "replicate":
        return init_device_mesh("cpu", (world, 1), mesh_dim_names=("rep", "dp"))
    return init_device_mesh("cpu", (2, world // 2), mesh_dim_names=("rep", "dp"))


def _model(mesh, mixed=False):
    torch.manual_seed(0)
    m = Net()
    for b in m.layers:
        fully_shard(b, mesh=mesh)
    if mixed:  # a plain parameter beside the DTensors (unused, so it stays equal on every rank)
        fully_shard(m.head, mesh=mesh)
        m.extra = torch.nn.Parameter(torch.ones(3))
    else:
        fully_shard(m, mesh=mesh)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.98), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    return m, opt, sched


def _batch(rank, u):
    g = torch.Generator().manual_seed(1000 * rank + u)
    x = torch.randn(8, 16, generator=g)
    return x, x.sum(-1) * 0.1


def _grad_norm(model):
    gs = [p.grad for p in model.parameters() if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs))).full_tensor().item()


def _wrap(opt, model, sched, root, **kw):
    args = dict(
        scheduler=sched,
        total_steps=TOTAL,
        goal="sharded test",
        every=5,
        sync_every=5,
        trial_steps=4,
        rank_check_every=10,
        backend="keep",
        summarize=False,
        verbose=False,
        run_dir=root / "r",
    )
    args.update(kw)
    return opal.wrap(opt, model, **args)


def _local_sha(model):
    import hashlib

    h = hashlib.sha256()
    for p in model.parameters():
        t = p.detach().to_local() if hasattr(p, "to_local") else p.detach()
        h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _replicas_agree(model):
    """Every rank's local shards against the other ranks holding the same shard (same replica key)."""
    import hashlib

    from opal.misc import sharding

    mine = []
    for p in model.parameters():
        t = sharding.local(p.detach()).contiguous()
        mine.append((sharding.replica_key(p).hex(), hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()))
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, mine)
    for i in range(len(mine)):
        seen = {}
        for rows in every:
            key, digest = rows[i]
            if seen.setdefault(key, digest) != digest:
                return False
    return True


def _worker(rank, world, folder, mesh_kind, case):
    torch.set_num_threads(1)
    root = Path(folder)
    dist.init_process_group(
        "gloo",
        init_method=(root / f"rendezvous-{case}").absolute().as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    out = dict(rank=rank, losses=[], plain=[], error=None, plain_gn={})
    try:
        mesh = _mesh(mesh_kind, world)
        m, o, s = _model(mesh, mixed=mesh_kind == "mixed")
        pm, po, ps = _model(mesh, mixed=mesh_kind == "mixed")
        start = 1
        if case == "resume_b":
            ck = torch.load(root / f"ck-r{rank}.pt", weights_only=False)
            w = _wrap(o, m, s, root, run_dir=root / "resumed")
            m.load_state_dict(ck["model"])
            w.load_state_dict(ck["opt"])
            pm.load_state_dict(ck["plain_model"])
            po.load_state_dict(ck["plain_opt"])
            ps.load_state_dict(ck["plain_sched"])
            start = ck["update"] + 1
        elif case == "example":  # installed by an agent at its first decision, as an agent would

            def agent(ws, msg, system):
                if "Decision 1." in msg:
                    (ws / "updater.py").write_text(EXAMPLE)
                return "installs the example"

            w = _wrap(o, m, s, root, backend=agent)
        else:
            w = _wrap(o, m, s, root)
        source = dict(
            equiv=EQUIV, diverge=DIVERGE, fail_first=FAIL_FIRST, resume_a=LOCAL_RULE, local=LOCAL_RULE
        ).get(case)
        if case == "skip":
            source = SKIP.replace("SKIP_AT", "6")
        for u in range(start, TOTAL + 1):
            x, y = _batch(rank, u)
            loss = torch.nn.functional.mse_loss(m(x), y)
            loss.backward()
            ploss = torch.nn.functional.mse_loss(pm(x), y)
            ploss.backward()
            out["plain_gn"][u] = _grad_norm(pm)
            if u == 3 and source is not None:
                w._apply(dict(k=99, requested_at=3, files=dict(w.files, **{"updater.py": source})), u)
            if u == 3 and case in ("analysis", "analysis_raise"):
                text = ANALYSIS.replace(
                    "RAISE", "if dist.get_rank() == 1:\n        raise ValueError('only on rank 1')" if case == "analysis_raise" else ""
                )
                w._apply(dict(k=99, requested_at=3, files=dict(w.files, **{"analysis.py": text})), u)
            if case == "repair" and u == 12 and rank == world - 1:  # one replica drifts outside any trial
                with torch.no_grad():
                    next(m.parameters()).to_local().add_(1.0)
            w.step(loss=loss)
            w.zero_grad()
            s.step()
            # a skipped update, and a trial rejected for replica divergence (rolled back without redoing the update),
            # leave the parameters and the optimizer state as they were: the reference skips that step too
            skipped = (case == "skip" and u == 6) or (case == "diverge" and mesh_kind != "shard" and u == 3)
            if not skipped:
                po.step()
            po.zero_grad()
            ps.step()
            out["losses"].append(loss.item())
            out["plain"].append(ploss.item())
            if case == "resume_a" and u == TOTAL // 2:
                torch.save(
                    dict(
                        model=m.state_dict(),
                        opt=w.state_dict(),
                        plain_model=pm.state_dict(),
                        plain_opt=po.state_dict(),
                        plain_sched=ps.state_dict(),
                        update=u,
                    ),
                    root / f"ck-r{rank}.pt",
                )
        out["version"] = w.version
        out["errors"] = [dict(e) for e in w.errors]
        out["analysis_off"] = bool(w._analysis_off)
        out["sha"], out["plain_sha"] = _local_sha(m), _local_sha(pm)
        out["replicas_agree"] = _replicas_agree(m)
        if rank == 0:
            out["system"] = (w.run.dir / "system.md").read_text() if (w.run.dir / "system.md").exists() else ""
    except Exception:
        import traceback

        out["error"] = traceback.format_exc()
    (root / f"{case}-r{rank}.json").write_text(json.dumps(out, default=str))
    dist.destroy_process_group()


def run(tmp_path, mesh_kind, case, world=None):
    world = world or (4 if mesh_kind == "hsdp" else 2)
    folder = tmp_path / f"{mesh_kind}-{case}"
    folder.mkdir(exist_ok=True)
    mp.spawn(_worker, args=(world, str(folder), mesh_kind, case), nprocs=world, join=True)
    outs = [json.loads((folder / f"{case}-r{r}.json").read_text()) for r in range(world)]
    for o in outs:
        assert o["error"] is None, o["error"]
    return folder, outs


def events(folder, name="r"):
    p = folder / name / "events.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def reports(folder):
    p = folder / "r" / "reports.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


@pytest.mark.parametrize("mesh_kind", ["shard", "replicate", "hsdp"])
def test_keep_matches_plain(tmp_path, mesh_kind):
    folder, outs = run(tmp_path, mesh_kind, "keep")
    for o in outs:
        assert o["losses"] == o["plain"]
        assert o["sha"] == o["plain_sha"]
        assert o["version"] == 0 and not o["errors"]
    # the starting analysis ran on every rank and reported global values
    reps = reports(folder)
    assert reps and all("grad_norm_total" in r["report"] for r in reps)
    for r in reps:
        assert r["report"]["grad_norm_total"] == pytest.approx(outs[0]["plain_gn"][str(r["update"])], rel=1e-4)
    system = outs[0]["system"]
    assert "SHARDED PARAMETERS (THIS RUN)" in system and "DTensor[" in system
    assert ("no replicas" in system) == (mesh_kind == "shard")


@pytest.mark.parametrize("mesh_kind", ["shard", "hsdp"])
def test_equivalent_swap_matches_plain(tmp_path, mesh_kind):
    folder, outs = run(tmp_path, mesh_kind, "equiv")
    for o in outs:
        assert o["losses"] == o["plain"]
        assert o["sha"] == o["plain_sha"]
        assert o["version"] == 1, o["errors"]  # the new updater passed its trial
    kinds = [e["kind"] for e in events(folder)]
    assert "rank_divergence" not in kinds and "ranks_resynced" not in kinds


@pytest.mark.parametrize("mesh_kind", ["replicate", "hsdp"])
def test_replica_divergence_is_rejected_exactly(tmp_path, mesh_kind):
    folder, outs = run(tmp_path, mesh_kind, "diverge")
    ev = events(folder)
    assert "rank_divergence" in [e["kind"] for e in ev]
    assert any(e["kind"] == "reverted" and e["update"] == 3 and e["restored"] is True for e in ev)
    for o in outs:
        assert o["losses"] == o["plain"]  # the trial was rolled back exactly; update 3 was not redone
        assert o["sha"] == o["plain_sha"]


def test_pure_sharding_has_no_replicas_to_compare(tmp_path):
    folder, outs = run(tmp_path, "shard", "diverge")
    kinds = [e["kind"] for e in events(folder)]
    assert "rank_divergence" not in kinds  # documented: with no replicas nothing is compared
    assert outs[0]["version"] == 1


@pytest.mark.parametrize("mesh_kind", ["shard", "hsdp"])
def test_first_update_failure_on_one_rank_is_undone_exactly(tmp_path, mesh_kind):
    folder, outs = run(tmp_path, mesh_kind, "fail_first")
    for o in outs:
        assert o["losses"] == o["plain"]
        assert o["sha"] == o["plain_sha"]
    reverted = [e for e in events(folder) if e["kind"] == "reverted"]
    assert reverted and reverted[0]["update"] == 3 and reverted[0]["restored"] is True


def test_skip_undoes_the_update_on_local_shards(tmp_path):
    folder, outs = run(tmp_path, "shard", "skip")
    for o in outs:
        assert o["losses"] == o["plain"]
        assert o["sha"] == o["plain_sha"]


def test_analysis_runs_on_every_rank_with_collectives(tmp_path):
    folder, outs = run(tmp_path, "shard", "analysis")
    reps = [r for r in reports(folder) if "grad_norm" in r["report"]]
    assert reps
    for r in reps:
        assert r["report"]["grad_norm"] == pytest.approx(outs[0]["plain_gn"][str(r["update"])], rel=1e-4)
    for o in outs:
        assert o["losses"] == o["plain"] and not o["analysis_off"]


def test_analysis_failure_on_another_rank_is_reported(tmp_path):
    folder, outs = run(tmp_path, "shard", "analysis_raise")
    assert any(e["where"] == "analysis.analyze" and "another rank" in e["error"] for e in outs[0]["errors"])
    for o in outs:
        assert o["losses"] == o["plain"]


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    folder, full = run(tmp_path, "shard", "resume_a")
    world = 2
    mp.spawn(_worker, args=(world, str(folder), "shard", "resume_b"), nprocs=world, join=True)
    resumed = [json.loads((folder / f"resume_b-r{r}.json").read_text()) for r in range(world)]
    for a, b in zip(full, resumed):
        assert b["error"] is None, b["error"]
        assert b["version"] == 1
        assert b["losses"] == a["losses"][TOTAL // 2 :]
        assert b["sha"] == a["sha"]


def test_fsdp1_flat_parameters_are_refused():
    m = torch.nn.Linear(2, 2)
    m.weight._is_flat_param = True
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="FSDP1"):
        opal.wrap(opt, m, total_steps=2, goal="t", backend=None, verbose=False, run_dir=None)



@pytest.mark.parametrize("mesh_kind", ["shard", "hsdp"])
def test_prompt_style_local_rule_runs_and_keeps_replicas_equal(tmp_path, mesh_kind):
    folder, outs = run(tmp_path, mesh_kind, "local")
    for o in outs:
        assert o["version"] == 1, o["errors"]
        assert not o["errors"] and o["replicas_agree"]
        assert all(math.isfinite(x) for x in o["losses"]) and o["losses"] != o["plain"]


@pytest.mark.parametrize("mesh_kind", ["shard", "hsdp"])
def test_the_example_updater_reproduces_the_starting_run(tmp_path, mesh_kind):
    folder, outs = run(tmp_path, mesh_kind, "example")
    for o in outs:
        assert o["version"] == 1, o["errors"]
        assert o["losses"] == o["plain"] and o["sha"] == o["plain_sha"]


def test_drifted_replica_is_repaired_from_its_group(tmp_path):
    folder, outs = run(tmp_path, "hsdp", "repair")
    resynced = [e for e in events(folder) if e["kind"] == "ranks_resynced"]
    assert resynced and resynced[0]["update"] == 20
    assert all(o["replicas_agree"] for o in outs)


def test_plain_parameters_beside_dtensors(tmp_path):
    folder, outs = run(tmp_path, "mixed", "keep")
    for o in outs:
        assert o["losses"] == o["plain"] and o["sha"] == o["plain_sha"] and o["replicas_agree"]
