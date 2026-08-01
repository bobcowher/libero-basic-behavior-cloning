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

Evaluation lives on `Agent` as a method. `Agent` already owns the model, the
device, and the task; evaluation needs exactly those. A separate evaluator module
was considered and rejected as premature — there is one policy implementation and
one caller, so the indirection would not have earned its existence yet.

The consequences of that choice are handled explicitly below rather than left to
discover later.

### `Agent.evaluate(n_eval=50, max_steps=600, env_num=1, seed=0) -> EvalResult`

Builds envs, sets init states, runs the loop above, counts successes, closes
envs, returns the result.

Returns a three-field dataclass:

```python
@dataclass
class EvalResult:
    success_rate: float
    successes: list[bool]              # per-rollout outcome
    steps_to_success: list[int | None] # None on failure
```

`n_eval` is `len(successes)`; wall time is the caller's business. A bare float
would cover today's headline number, but the two lists cost one line each and are
what any later failure analysis reads.

### Lazy construction

`Agent.__init__` currently builds a DataLoader and an env eagerly. Both become
lazy properties:

- `Agent.dl` — built on first access. Scoring a checkpoint must not construct the
  training dataset; at `hdf5_cache_mode="low_dim"` that is a multi-second load,
  and at `"all"` it is gigabytes of RAM for data evaluation never reads.
- `Agent.env` — built on first access. Training reads from the dataloader and
  never steps an env, so training runs currently pay for a MuJoCo context they do
  not use.

This is the load-bearing part of keeping evaluation on `Agent`. Without it,
`eval.py` builds the training set to run rollouts.

### Observation preprocessing

`Agent._preprocess_obs(raw_obs) -> (images, proprio)` is the **single** live-env
preprocessing path. Per CONTEXT.md, HDF5-vs-live key mismatch is the top source
of train/eval skew and it fails silently, so there must be exactly one
implementation of this translation.

- image: `agentview_image`, HWC uint8 `[0,255]` → CHW float32 `[0,1]`.
  **No vertical flip** — stored demos and the live env share the `opengl`
  convention.
- proprio: `concat(robot0_joint_pos (7,), robot0_gripper_qpos (2,))` → `(9,)`,
  raw and unnormalized, because the training loader delivers it raw.

If proprio normalization is ever introduced in training, it must be applied here
with identical per-dimension statistics.

Vector envs return an array of observation dicts, so this handles a batch and
returns batched tensors.

### Env construction

Mirrors `metric.py:70-94`:

```python
env_args = {
    "bddl_file_name": <bddl path via get_libero_path()>,
    "camera_heights": 128,
    "camera_widths": 128,
}
```

Paths are always built through `get_libero_path()`. The `env_args` recorded in
the HDF5 file contain stale `chiliocosm/` paths and must be ignored.

### Parallelism

`DummyVectorEnv` at `env_num == 1`, which is what LIBERO's own evaluation uses in
the single-env case. This is serial execution — no subprocesses, no fork-safety
or EGL-in-worker debugging — while the batched interface comes from library code
rather than hand-rolled code.

Moving to parallel evaluation later is a single constructor change to
`SubprocVectorEnv`, with no caller changes because the interface is identical.
Serial evaluation is the known bottleneck on the only metric that matters, so
this swap is expected, but it happens after a known-good serial baseline exists
to compare against.

### `task_id` becomes a parameter

`Agent.__init__` takes `task_id=0` and derives the dataset filename from
`task_suite.get_task_demonstration(task_id)` rather than hardcoding both. This is
not speculative generality: both values are already hardcoded constants at
`agent.py:28` and `agent.py:33`, and deriving one from the other removes the
possibility of an `Agent` whose dataset and env disagree about which task it is.

Multi-task evaluation is out of scope. When it arrives, the shape of the problem
is "one policy, N task ids," which does not fit an object whose identity is a
single task — expect to revisit this choice then, deliberately, rather than
discovering it mid-change.

## Changes to existing files

**`agent.py`** — gains `evaluate()`, `_preprocess_obs()`, lazy `dl` / `env`
properties, and a `task_id` parameter. `test()` is deleted along with the
`ControlEnv(has_renderer=True)` branch. `train()` gains an optional `eval_every`
that calls `self.evaluate()`, so the training and evaluation paths cannot drift.

**`test.py`** — deleted. Replaced by `eval.py`.

**`eval.py`** — new, thin CLI:

```
python eval.py --ckpt checkpoints/bc_network --task 0 --n-eval 50
```

Constructs an `Agent`, loads the checkpoint, calls `evaluate()`, prints a
one-line summary.

**`env_wrapper.py`** — deleted. `LiberoObsWrapper` is superseded by
`Agent._preprocess_obs`, and vector envs return arrays of dicts rather than
single dicts. Its only consumers were `Agent.__init__` (which wraps an env solely
to print observation shapes) and the deleted `Agent.test()`. Keeping it would
leave two live-obs preprocessing paths, which is precisely the skew this design
is trying to prevent.

**`build.sh`** — the run line points at `eval.py`.

### On the deleted viewer path

`Agent.test()` calls `env.render()` on a `ControlEnv(has_renderer=True)`. Under
`MUJOCO_GL=egl` this cannot work: per CONTEXT.md the viewer requires `glfw` and
provides no camera observations, so it cannot coexist with the offscreen
rendering that produces policy inputs. Watching a policy means writing MP4s of
`agentview_image`, which mirrors real robot deployment. The viewer is a toy and
does not justify a code path.

## Error handling

- **Env construction** — `metric.py` retries construction up to five times with a
  five-second sleep, working around an intermittent frame-buffer issue. The same
  retry is reproduced; this failure is real and environment-specific.
- **Env teardown** — envs are closed in a `finally` block. An exception mid-eval
  must not leak an env and produce EGL tracebacks at interpreter shutdown, which
  is the established cause of that symptom in this project.
- **Checkpoint missing** — fail immediately with the attempted path, rather than
  evaluating a randomly initialized network and reporting 0%.
- **Action shape** — assert the policy returns `(env_num, 7)` before the first
  `env.step`. A silently wrong action shape is otherwise indistinguishable from a
  bad policy.

## Testing

The harness is itself test infrastructure, so verification is behavioral rather
than a large unit-test suite:

1. **Zero-action smoke run** — a few rollouts with an all-zeros action must
   return 0% without crashing. Exercises the loop, env lifecycle, counting, and
   preprocessing shapes in one pass, with no trained model required.
2. **Lazy-construction check** — `evaluate()` runs without the DataLoader ever
   being constructed. This is the property the lazy design exists to provide, so
   it is asserted rather than assumed.
3. **Known-checkpoint run** — evaluate the existing trained checkpoint. Any
   result is informative: it establishes the current baseline.

Shape and determinism checks fold into (1) and (3) rather than standing alone;
this is test infrastructure, not a library with external consumers.

## Success criteria

`python eval.py --ckpt <path>` prints a success rate over 50 rollouts from
benchmark init states, with no EGL tracebacks, without building the training
dataset, and repeating identically at a fixed seed.

## Notes for later

- Target for the planned ACT baseline is 70-90% on a single LIBERO-Spatial task.
  At 30%, the problem is the data pipeline or normalization, not architecture.
- Failure taxonomy for this task, once diagnostics are built: (a) never reaches
  the bowl, (b) reaches but fumbles the grasp, (c) grasps but drops in transit.
  `successes` and `steps_to_success` on `EvalResult` are the raw material those
  buckets derive from.
- Action-chunking policies (the planned ACT baseline predicts 16 actions and
  executes 8) need per-rollout state reset between episodes. Today's policy is
  stateless, so no reset hook is built; adding one is a single call at the top of
  each rollout.
