# LIBERO Behavior Cloning — Context Handoff

## Goal

Build a single-task behavior-cloning policy on LIBERO, then extend to language
conditioning (a "simple VLA"). Long-term target is an SO-101 arm via LeRobot.

Deliberate sequencing: **BC first, language last.** A VLA is conditioned
behavior cloning — the architecture is the easy part, the data is not.

## Background

Strong in RL (SAC, Q-learning, DreamerV2/V3 from scratch, world models,
behavior cloning via SAC bootstrapping). **No prior pure BC experience.**
PyTorch, Python. Hardware: RTX 5090 (32GB).

Three RL instincts that mislead in BC:

1. **Validation loss is nearly uncorrelated with success rate.** Rollout-based
   eval is the only real metric. 50 rollouts from fixed init states; anything
   under 20 is noise.
2. **Compounding error is the whole problem.** Off-distribution states spiral.
   Demo coverage beats model capacity.
3. **Multimodality breaks naive regression.** MSE learns the mean of a bimodal
   demonstrator. This is *why* ACT (CVAE) and Diffusion Policy exist.

## Environment status — WORKING

Conda env `libero`, Python 3.8, robosuite 1.4 (pinned — 1.5 removes
`SingleArmEnv` and breaks LIBERO). Repo at `~/pythonprojects/LIBERO`.

Env vars in `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`:

```
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
ROBOSUITE_LOG_LEVEL=ERROR
PYTHONWARNINGS=ignore::UserWarning,ignore::DeprecationWarning
```

Resolved issues:

- EGL shutdown tracebacks were an unclosed env. `env.close()` fixed it.
- `gym` NumPy 2.0 notice is a bare `print`, not a warning — no filter catches
  it. Line was removed from site-packages. `numpy<2` is pinned.
- robosuite macros: `setup_macros.py` has been run.

Env classes available (this install): `DummyVectorEnv`, `OffScreenRenderEnv`,
`SegmentationRenderEnv`, `SubprocVectorEnv`. **No `OnScreenRenderEnv`** —
viewer requires `ControlEnv` from `libero.libero.envs.env_wrapper` directly.

Two rendering modes, one per process:

| | `MUJOCO_GL` | flags | obs images? |
|---|---|---|---|
| Offscreen (default, deployment-faithful) | `egl` | `OffScreenRenderEnv` | yes |
| Viewer (debug only) | `glfw` | `ControlEnv(has_renderer=True, has_offscreen_renderer=False, use_camera_obs=False)` | no |

Watching = write MP4s of `obs["agentview_image"]`. This mirrors real-robot
deployment; the viewer is a toy.

`SubprocVectorEnv` is available and worth wiring early — serial eval is the
bottleneck on the only metric that matters.

## Dataset — schema verified

`libero_spatial`, task 0: *"pick up the black bowl between the plate and the
ramekin and place it on the plate"*. 50 demos, 5068 transitions, **~101 steps
per episode**.

```
data.attrs: num_demos=50, total=5068, macros_image_convention='opengl'
            problem_info -> JSON with "language_instruction"
            env_args     -> JSON (contains STALE chiliocosm/ paths — ignore,
                            always build paths via get_libero_path())
data/demo_N.attrs: num_samples, init_state, model_file
data/demo_N/actions              (T,7)   OSC_POSE deltas, 3 pos + 3 axis-angle + 1 gripper
data/demo_N/dones, rewards       (T,)
data/demo_N/robot_states         (T,9)   == joint_states + gripper_states
data/demo_N/states               (T,92)  full MuJoCo state
data/demo_N/obs/agentview_rgb    (T,128,128,3) uint8
data/demo_N/obs/eye_in_hand_rgb  (T,128,128,3) uint8
data/demo_N/obs/joint_states     (T,7)   float64
data/demo_N/obs/gripper_states   (T,2)   float64
data/demo_N/obs/ee_pos, ee_ori, ee_states  (redundant with the above)
```

### Gotchas

- **Key mismatch, HDF5 vs live env** — the top source of train/eval skew:

  | HDF5 | live env |
  |---|---|
  | `agentview_rgb` | `agentview_image` |
  | `eye_in_hand_rgb` | `robot0_eye_in_hand_image` |
  | `joint_states` | `robot0_joint_pos` |
  | `gripper_states` | `robot0_gripper_qpos` |

- **No vertical flip for training.** Stored images use the `opengl` convention
  and so does the live env (robosuite `IMAGE_CONVENTION`). They agree. Flip
  only when writing video for human viewing.
- **Gripper action (dim 6) is bimodal in {-1,+1}.** Do not standardize it with
  the same statistics as the continuous dims — it smears toward zero and
  presents as "the gripper never fully closes."
- **Proprio scale gap**: joints ±2.45 rad, gripper ±0.039. Per-dimension
  normalization required; a global scalar erases the gripper.
- **Language is per-file**, in `data.attrs["problem_info"]`, not per-demo.

## Data loading — decision made

Two options were built/evaluated:

1. `libero_dataset.py` (hand-rolled, ~200 lines, **untested**) — readable,
   uint8 images, explicit padding and masking.
2. **LIBERO's `libero.lifelong.datasets.get_dataset`** ← chosen, since it's
   battle-tested and unread library code beats unread generated code.

```python
import libero.lifelong.datasets as D

obs_modality = {
    "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
    "low_dim": ["joint_states", "gripper_states"],
}
dataset, shape_meta = D.get_dataset(
    dataset_path=hdf5_path,
    obs_modality=obs_modality,
    initialize_obs_utils=True,
    seq_len=16,
    frame_stack=1,
    hdf5_cache_mode="low_dim",
)
```

Signature: `(dataset_path, obs_modality, initialize_obs_utils=True, seq_len=1,
frame_stack=1, filter_key=None, hdf5_cache_mode='low_dim', *args, **kwargs)`

`SequenceVLDataset(sequence_dataset, task_emb)` is trivial — it only staples
`task_emb` onto each sample. Skip it for single-task; use it for language
conditioning later.

### Verified output shapes

```
actions                (16, 7)
obs/agentview_rgb      (16, 3, 128, 128) float32, [0,1], CHW
obs/eye_in_hand_rgb    (16, 3, 128, 128) float32, [0,1], CHW
obs/joint_states       (16, 7)  float32, RAW (radians, ±2.45)
obs/gripper_states     (16, 2)  float32, RAW (±0.039)
```

### Findings from that output

- **`seq_len` applies to observations too**, despite `frame_stack=1`. Robomimic
  is built for BC-RNN. For ACT-style, take `obs[:, 0]` and keep all 16 actions.
  Costs 16x redundant image reads — mitigate with `hdf5_cache_mode="all"`
  (~500MB/task, single-task only) or accept it.
- **`ObsUtils` handles image conversion via GLOBAL state.** The eval script
  *must* call `get_dataset` or
  `ObsUtils.initialize_obs_utils_with_obs_specs` with the identical
  `obs_modality` dict, or rollout images get processed differently than
  training images. Silent failure mode.
- **Low-dim is NOT normalized.** Action normalization still unverified —
  check `actions.min()/max()`.

## Immediate next steps

1. **Print `actions.min()/max()`.** Determine whether normalization is needed
   (expect raw, in [-1,1]).
2. **Demo replay round-trip test — THE GATE.** Pull windows for demo 0 in
   order, reconstruct the action sequence, `set_init_state` from the demo's
   init state, step through the env, confirm `done` fires. If a scripted-perfect
   sequence from the loader doesn't succeed, the pipeline is misaligned and
   nothing downstream is meaningful. **Do not write model code before this
   passes.**
3. Build the `SingleObsChunk` adapter (obs[:,0] + 16 actions).
4. Wire `SubprocVectorEnv` eval harness: 50 rollouts from the task's fixed
   init states, success rate as the headline metric.

## Then: the model

Plain ACT-minus-CVAE baseline:

- ResNet18, ImageNet init, **unfrozen** (ImageNet features are semantic and
  pose-invariant; manipulation needs geometric precision — fine-tuning wins
  at this scale; 20k image-action pairs is plenty for 11M params)
- **Spatial softmax instead of global average pooling** — biggest single win
  for visuomotor policies; GAP destroys the spatial layout you need
- **GroupNorm instead of BatchNorm** — BN's running stats interact badly with
  EMA and small batches
- **Separate encoder weights per camera** — wrist and agentview see very
  different distributions
- Concat with proprio (9-dim), 3-4 transformer blocks
- Predict 16 actions, L1 loss, execute 8 then replan

Wrist camera does most of the work (gripper-frame, position-invariant contact
detail); agentview provides scene context.

**Target: 70-90% success on a single LIBERO-Spatial task.** At 30%, the
problem is the data pipeline or normalization, not the architecture — do not
add a diffusion head before ruling those out.

Failure taxonomy for this task: (a) never reaches the bowl, (b) reaches but
fumbles the grasp, (c) grasps but drops in transit. Each points somewhere
different.

## After the baseline

Language conditioning is ~30 lines: encode the instruction to a pooled vector,
concat into the conditioning stack. LIBERO-Spatial is the right testbed —
identical scenes, instructions that genuinely disambiguate.

**Critical control:** compare a text embedding against a task-ID one-hot. If
they perform the same, the language pathway is doing nothing and the policy is
inferring the task from the scene. This is the failure mode most hobby VLAs
ship with unnoticed.

## Reference implementations

- `openvla/openvla` → `experiments/robot/libero/run_libero_eval.py` — the most
  cloned eval harness; `libero_utils.py` has preprocessing;
  `regenerate_libero_dataset.py` for re-rendering
- `huggingface/lerobot-libero` — LIBERO in LeRobot format, matching the
  eventual SO-101 pipeline
- `keivalya/mini-vla` — ~150 LOC core model, readable end to end
- SmolVLA (HuggingFace) — SO-101-native, reduced visual tokens, action expert

## Open corrections to earlier assumptions

- Demos **do** ship with rendered images; regeneration scripts exist for
  filtering/re-resolution, not because images are missing.
- Episodes are ~101 steps, not 150-250.
- "Frozen ResNet18" was wrong — that advice belonged to the large-VLM case,
  not an 11M-param backbone on 20k samples.
