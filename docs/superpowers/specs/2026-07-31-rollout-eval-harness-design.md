# Rollout Evaluation Harness — Design

**Date:** 2026-07-31
**Status:** Approved, pending implementation plan

## Purpose

Produce a trustworthy success-rate number for a trained BC policy on a LIBERO
task, before the project grows to multiple tasks.

Validation loss is nearly uncorrelated with success rate in behavior cloning.
Rollout success rate is the only real metric, and there is currently no way to
measure it. Every downstream decision — architecture changes, normalization
fixes, adding tasks — is unmeasurable until this exists.

## Scope

**In scope:** rollout evaluation producing a success rate.

**Explicitly out of scope** (deliberate YAGNI, revisit when needed):

- Run tracking / cross-run comparison persistence
- Failure diagnostics: video dumps, failure taxonomy bucketing
- Pipeline correctness gates (demo replay round-trip, normalization audits)

The design leaves a seam for the replay gate — a policy that emits recorded demo
actions satisfies the same `BCPolicy` interface — but that gate is not built here.

## Protocol

Replicated from LIBERO's own evaluation loop, `libero/lifelong/metric.py:102-161`.
These details are non-obvious and fail silently if guessed:

1. `env.reset()`
2. `env.set_init_state(init_states[indices])`, indices wrapping with
   `% init_states.shape[0]`
3. **Five steps of zero actions** to settle physics before the policy observes
   anything. Skipping this means the first observation shows a non-settled arm
   that does not match any training frame.
4. Step until `max_steps`, accumulating **sticky dones**:
   `dones[k] = dones[k] or done[k]`. Success means `done` fired at any point,
   not that it was set on the final step.
5. Break early once all envs are done.
6. `success_rate = num_success / n_eval`

### Initial states

`task_suite.get_task_init_states(task_id)` — the standard LIBERO protocol, used
by published results. Returns a tensor loaded from the benchmark's init-states
file (`benchmark/__init__.py:158`).

These are not the states the demonstrations start from, so this measures
generalization rather than memorization. A "can the policy fit its own training
data" check would use the HDF5 per-demo `init_state` attributes instead; that is
out of scope here.

### Defaults

| Parameter | Value | Rationale |
|---|---|---|
| `n_eval` | 50 | Project standard. LIBERO ships 20; fewer than 20 rollouts is noise. |
| `max_steps` | 600 | LIBERO default. Demonstrations run ~101 steps, so this is generous. |
| `env_num` | 1 | Serial. See parallelism below. |
| `seed` | 0 | Reproducibility. |

## Architecture

Three new modules with one responsibility each, plus targeted edits to existing
files.

### `policy.py` — `BCPolicy`

Owns the model **and** observation preprocessing. Preprocessing must match
training exactly; training is the policy's concern, so the two live together and
train/eval skew has one place to hide instead of three.

```python
class BCPolicy:
    def __init__(self, model, device): ...
    def reset(self): ...                    # no-op; hook for ACT action chunking
    def act(self, raw_obs) -> np.ndarray:   # (N, 7)
```

`raw_obs` is the vector-env output: an array of raw LIBERO observation dicts.
`act` handles key translation, image conversion, batching, and the tensor
round-trip.

Preprocessing, matching `env_wrapper.py` and the training loader:

- image: `agentview_image`, HWC uint8 `[0,255]` → CHW float32 `[0,1]`.
  **No vertical flip** — stored demos and the live env share the `opengl`
  convention.
- proprio: `concat(robot0_joint_pos (7,), robot0_gripper_qpos (2,))` → `(9,)`,
  raw and unnormalized, because the training loader delivers it raw.

If proprio normalization is ever introduced in training, it must be applied here
with identical per-dimension statistics.

`reset()` exists so action-chunking policies (the planned ACT baseline, which
predicts 16 actions and executes 8) can clear their buffer between rollouts
without changing the evaluator.

### `evaluate.py` — `evaluate(...) -> EvalResult`

```python
def evaluate(policy, task_id=0, n_eval=50, max_steps=600,
             env_num=1, seed=0) -> EvalResult
```

Owns envs, init states, the step loop, and success counting. Knows nothing about
`Agent`, the dataloader, or optimizers. Any callable satisfying the `BCPolicy`
interface can be evaluated.

Environment construction mirrors `metric.py:70-94`:

```python
env_args = {
    "bddl_file_name": <bddl path via get_libero_path()>,
    "camera_heights": 128,
    "camera_widths": 128,
}
```

Paths are always built through `get_libero_path()`. The `env_args` recorded in
the HDF5 file contain stale `chiliocosm/` paths and must be ignored.

`EvalResult` is a dataclass:

| Field | Type | Meaning |
|---|---|---|
| `success_rate` | `float` | `num_success / n_eval` |
| `n_eval` | `int` | Rollouts run |
| `successes` | `list[bool]` | Per-rollout outcome |
| `steps_to_success` | `list[int \| None]` | Steps until `done`, `None` on failure |
| `wall_time` | `float` | Seconds elapsed |

The evaluator closes its envs before returning. Unclosed envs were the
established cause of EGL shutdown tracebacks in this project.

### `eval.py` — CLI

```
python eval.py --ckpt checkpoints/bc_network --n-eval 50
```

Loads the model, wraps it in `BCPolicy`, calls `evaluate`, prints a one-line
summary. No dataloader is constructed, so running an evaluation does not build
the multi-gigabyte dataset cache.

### Parallelism

`DummyVectorEnv` at `env_num == 1`, which is what LIBERO's own evaluation uses
in the single-env case. This is serial execution — no subprocesses, no
fork-safety or EGL-in-worker debugging — while the batched interface comes from
library code rather than hand-rolled code.

Moving to parallel evaluation later is a single constructor change to
`SubprocVectorEnv`. No caller changes, because the interface is identical. Serial
evaluation is the known bottleneck on the only metric that matters, so this swap
is expected, but it happens after a known-good serial baseline exists to compare
against.

## Changes to existing files

**`agent.py`** — `test()` is deleted. `Agent` exposes `self.policy` (a
`BCPolicy` wrapping its model) and `train()` gains an optional `eval_every`
parameter that calls the same `evaluate()` function. The evaluation path and the
training path therefore cannot drift.

**`test.py`** — deleted, along with the `ControlEnv(has_renderer=True)` branch in
`Agent.__init__`. That path calls `env.render()` under `MUJOCO_GL=egl`, which
cannot work: the viewer requires `glfw` and provides no camera observations.
Watching a policy means writing MP4s of `agentview_image`, which mirrors real
robot deployment. The viewer is a toy and is not worth a code path.

**`env_wrapper.py`** — deleted. `LiberoObsWrapper` is superseded: preprocessing
moves into `BCPolicy`, and vector envs return arrays of observation dicts rather
than single dicts. Its only consumers were `Agent.__init__` (which wraps an env
solely to print observation shapes) and the deleted `Agent.test()`. The training
loop reads from the dataloader and never touches a live env. Removing it keeps
preprocessing in exactly one place, which is the entire point of putting it in
`BCPolicy`.

Consequently `Agent.__init__` stops constructing an env at all. Training does not
need one, and building an env per `Agent` makes training startup pay for a
MuJoCo context it never uses.

**`build.sh`** — the run line points at `eval.py`.

## Error handling

- **Env construction** — `metric.py` retries construction up to five times with a
  five-second sleep, working around an intermittent frame-buffer issue. The same
  retry is reproduced; this failure is real and environment-specific.
- **Env teardown** — envs are closed in a `finally` block. An exception mid-eval
  must not leak an env and produce EGL tracebacks at interpreter shutdown.
- **Checkpoint missing** — fail immediately with the attempted path, rather than
  evaluating a randomly initialized network and reporting 0%.
- **Action shape or range** — assert the policy returns `(env_num, 7)` before the
  first `env.step`. A silently wrong action shape is otherwise indistinguishable
  from a bad policy.

## Testing

The harness is itself test infrastructure, so verification is behavioral rather
than a large unit-test suite:

1. **Zero-action policy** — a policy emitting all zeros must produce a 0% success
   rate without crashing. Exercises the full loop, env lifecycle, and counting.
2. **Shape assertions** — preprocessed image is `(N, 3, 128, 128)` float32 in
   `[0, 1]`; proprio is `(N, 9)` float32.
3. **Determinism** — two runs at the same seed with the same checkpoint produce
   an identical success rate.
4. **Known-checkpoint run** — evaluate the existing trained checkpoint. Any
   result is informative: it establishes the current baseline.

## Success criteria

`python eval.py --ckpt <path>` prints a success rate over 50 rollouts from
benchmark init states, with no EGL tracebacks, and repeats identically at a fixed
seed.

## Notes for later

- Target for the planned ACT baseline is 70-90% on a single LIBERO-Spatial task.
  At 30%, the problem is the data pipeline or normalization, not architecture.
- Failure taxonomy for this task, once diagnostics are built: (a) never reaches
  the bowl, (b) reaches but fumbles the grasp, (c) grasps but drops in transit.
  `steps_to_success` and per-rollout outcomes are already carried by
  `EvalResult`, which is the data those buckets would be derived from.
