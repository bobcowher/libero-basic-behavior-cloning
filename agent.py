import contextlib
import io
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dataloader import DataLoader
from model import Model

# libero pulls in `gym`, which prints an unmaintained-package notice straight
# to stderr on import (gym_notices) instead of raising a warnings.UserWarning,
# so PYTHONWARNINGS can't filter it. Swallow stderr just for this import.
with contextlib.redirect_stderr(io.StringIO()):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import (OffScreenRenderEnv, DummyVectorEnv,
                                    SubprocVectorEnv)
    from libero.libero.envs.env_wrapper import ControlEnv

from torch.utils.tensorboard import SummaryWriter

# Rollout outcomes are chaotic in the low-order bits: a tiny action difference
# compounds over hundreds of closed-loop steps into success vs failure. cuDNN
# TF32 is ON by default on Ampere+ and makes conv output depend on BATCH SIZE,
# so evaluate(env_num=10) and evaluate(env_num=1) scored different scenes with
# identical weights. Measured on this model: max action delta between batch-1
# and batch-10 is 1.64e-4 with TF32, 1.49e-8 without.
#
# Set at import, before any model is built, so every entry point gets it.
# Costs little here -- the net is tiny and training is not GPU-bound.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True

IMG_SIZE = 128
ACTION_DIM = 7

# Steps of zero action after set_init_state, before the policy sees anything.
# Copied from LIBERO's own eval loop (lifelong/metric.py). Without it the first
# observation shows a mid-transient arm that matches no training frame.
SETTLE_STEPS = 5


class WatchEnv(ControlEnv):
    """OffScreenRenderEnv plus an on-screen window.

    Deliberately a copy of OffScreenRenderEnv (env_wrapper.py:152) with
    has_renderer flipped, because that class hardcodes it to False. Watching
    and scoring must differ by exactly this one boolean and nothing else --
    same class, same kwargs, same physics -- so a rollout you watch is the
    rollout eval.py scores.

    Offscreen rendering stays ON: it produces the agentview_image the policy
    consumes. The window is an extra view, not the policy's input.
    """

    def __init__(self, **kwargs):
        kwargs["has_renderer"] = True
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)

    def render(self, **kwargs):
        # ControlEnv defines no render(); the window belongs to the robosuite
        # env it wraps. The vector env calls worker.render() -> env.render(),
        # so this forward is what lets the shared rollout loop drive the view.
        return self.env.render(**kwargs)


@dataclass
class EvalResult:
    scenes: list             # benchmark init-state index per rollout
    successes: list          # per-rollout bool
    steps_to_success: list   # step index of success, None on failure

    @property
    def success_rate(self):
        return sum(self.successes) / len(self.successes)

    @property
    def solved(self):
        """Scene indices that succeeded -- which ones, not just how many."""
        return [s for s, ok in zip(self.scenes, self.successes) if ok]

    def __str__(self):
        solved = [s for s in self.steps_to_success if s is not None]
        tail = f", median {int(np.median(solved))} steps" if solved else ""
        return (f"success {self.success_rate:.1%} "
                f"({sum(self.successes)}/{len(self.successes)}){tail}")


class Agent:

    def __init__(self, task_id=0, lr=1e-4, device=None,
                 ckpt="checkpoints/bc_network", benchmark_name="libero_spatial"):
        self.task_id = task_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.task_suite = benchmark.get_benchmark_dict()[benchmark_name]()
        self.task = self.task_suite.get_task(task_id)

        self.model = Model(
            input_shape=(3, IMG_SIZE, IMG_SIZE),
            num_actions=ACTION_DIM,
            hidden_dim=256,
            checkpoint_dir=os.path.dirname(ckpt) or ".",
            name=os.path.basename(ckpt),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._dl = None

    # ---- lazy dataset ----------------------------------------------------
    # Scoring a checkpoint must not construct the training set. Built on first
    # access so eval-only runs never pay for it.
    @property
    def dl(self):
        if self._dl is None:
            # Derived from the task, so the dataset and the env can never
            # disagree about which task this Agent is.
            demo_path = self.task_suite.get_task_demonstration(self.task_id)
            self._dl = DataLoader(dataset_filename=demo_path, device=self.device)
        return self._dl

    def load_checkpoint(self):
        path = self.model.checkpoint_file
        if not os.path.exists(path):
            # Loud, not a 0% score from a randomly initialized network.
            raise FileNotFoundError(f"no checkpoint at {path}")
        self.model.load_checkpoint()

    # ---- observation preprocessing ---------------------------------------
    def _preprocess_obs(self, raw_obs):
        """Array of live LIBERO obs dicts -> (images, proprio) device tensors.

        The ONLY live-env preprocessing path. HDF5 and live envs use different
        keys (agentview_rgb vs agentview_image, joint_states vs
        robot0_joint_pos), which is the top source of silent train/eval skew --
        so there is exactly one implementation of the translation.

        No vertical flip: stored demos and the live env share the opengl
        convention. Proprio is raw because the training loader delivers it raw;
        if training ever normalizes it, normalize here with the SAME per-dim
        stats.
        """
        images = np.stack([o["agentview_image"] for o in raw_obs])   # (N,H,W,3) uint8
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        proprio = np.stack([
            np.concatenate([o["robot0_joint_pos"], o["robot0_gripper_qpos"]])
            for o in raw_obs
        ]).astype(np.float32)                                        # (N, 9)

        return (torch.from_numpy(images).to(self.device),
                torch.from_numpy(proprio).to(self.device))

    @torch.no_grad()
    def _act(self, raw_obs):
        """Actions for a list of live obs dicts.

        Evaluated ONE observation at a time even when several envs are in
        flight. Batched conv reductions are batch-size dependent -- ~1e-4 with
        cuDNN TF32, ~1e-8 without -- and closed-loop rollouts amplify either
        into a different success/failure outcome. That made evaluate() report
        different solved scenes at env_num=1 and env_num=10.

        A policy is a pure function of one observation; batching it is an
        optimization that must not change the answer. Costs nothing that
        matters here: the MuJoCo step dominates wall clock, not this net.
        """
        images, proprio = self._preprocess_obs(raw_obs)
        return np.stack([
            self.model(images[i:i + 1], proprio[i:i + 1])[0].cpu().numpy()
            for i in range(images.shape[0])
        ])

    def _init_states(self):
        """The benchmark's fixed eval init states.

        Replicates benchmark.get_task_init_states (benchmark/__init__.py:158)
        instead of calling it: that helper calls torch.load without
        weights_only, and torch >= 2.6 defaults weights_only=True, which
        rejects the numpy pickle. weights_only=False is safe here -- the file
        is local benchmark data shipped with the LIBERO install, not a
        downloaded checkpoint.
        """
        path = os.path.join(get_libero_path("init_states"),
                            self.task.problem_folder, self.task.init_states_file)
        return torch.load(path, weights_only=False)

    # ---- env -------------------------------------------------------------
    def _build_env(self, env_num, seed, render=False):
        bddl = os.path.join(get_libero_path("bddl_files"),
                            self.task.problem_folder, self.task.bddl_file)
        env_args = {"bddl_file_name": bddl,
                    "camera_heights": IMG_SIZE, "camera_widths": IMG_SIZE}

        if render:
            # A window per worker is unwatchable, and the on-screen renderer is
            # not process-safe under spawn.
            assert env_num == 1, "render requires env_num=1"
            # The viewer camera only. It does not touch camera_names, so the
            # policy's agentview_image is unaffected -- you are watching the
            # same frames the net is fed.
            env_args["render_camera"] = "agentview"
            make = lambda: WatchEnv(**env_args)          # noqa: E731
        else:
            make = lambda: OffScreenRenderEnv(**env_args)  # noqa: E731

        cls = DummyVectorEnv if env_num == 1 else SubprocVectorEnv

        if env_num > 1:
            # MUST be spawn, not fork. The parent has already initialized CUDA
            # (the model lives on GPU), and NVIDIA driver state does not
            # survive fork() -- every forked worker dies with
            # EGL_BAD_ALLOC in eglCreateContext. Spawn gives each worker a
            # fresh interpreter that initializes its own EGL display.
            #
            # Viable because LIBERO wraps env fns in CloudpickleWrapper
            # (venv.py:41), so the lambda below pickles. Requires every entry
            # point to guard with `if __name__ == "__main__"`, or spawned
            # children re-execute it.
            mp.set_start_method("spawn", force=True)

        # LIBERO's own eval retries here: env creation intermittently fails on
        # a frame buffer error (lifelong/metric.py:85-100).
        for attempt in range(5):
            try:
                env = cls([make for _ in range(env_num)])
                # A LIST, not an int. BaseVectorEnv.seed(int) expands to
                # [seed + i for i in range(env_num)] (venv.py:849), so worker k
                # runs under seed k -- and a scene's rollout then depends on
                # which worker happened to draw it, i.e. on env_num. Passing a
                # list is taken verbatim (venv.py:851), so every worker is
                # identical and a scene scores the same at any env_num.
                env.seed([seed] * env_num)
                return env
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5)
        raise RuntimeError("unreachable: env creation loop exhausted")

    # ---- watch ------------------------------------------------------------
    # Scenes watched by default: a stride across the whole 50-scene set rather
    # than the first N. Scenes 0-4 are a corner of the set; 0,10,20,30,40
    # samples all of it, so what you watch is representative of what eval.py
    # scores.
    WATCH_SCENES = list(range(0, 50, 10))

    def test(self, scenes=None, max_steps=300, reference_protocol=False):
        """Watch the policy run, in a window, on the scenes eval.py scores.

        This is a VIEW of the eval, not a separate thing: every rollout goes
        through evaluate(), so a scene shown succeeding here is a scene counted
        succeeding there. test() and evaluate() previously kept their own
        rollout loops and disagreed about the same checkpoint -- there is now
        one loop, and the only difference between watching and scoring is the
        has_renderer boolean in WatchEnv.

        Needs MUJOCO_GL=glfw and a display -- test.py sets that before this
        module is imported, since mujoco picks its backend at import time.

        Runs one scene per evaluate() call so outcomes print as they happen
        instead of after the last rollout. Each rollout still gets a fresh env
        (see evaluate), which means the window is torn down and reopened
        between scenes -- correctness over a persistent window.

        max_steps defaults to 300, well under the 600 eval.py uses: successful
        rollouts finish near the demos' ~101 steps, so nothing real is
        truncated, and a failure stops wasting your time sooner. Rendering
        makes failures the expensive ones.

        reference_protocol=True instead reproduces the reference project's odd
        setup once: seed 0, **two** env.reset() calls, no init state, no settle
        steps. Kept because it is the known-good comparison, but it is NOT a
        scored scene -- reset() samples fresh placements, a third distribution
        that appears in neither the demos nor the 50 eval states. Not the
        default for exactly that reason.
        """
        if reference_protocol:
            r = self.evaluate(scenes=[None], max_steps=max_steps, render=True)
            print("reference protocol (unscored scene): "
                  f"{'ok' if r.successes[0] else 'FAIL'}")
            return r

        scenes = self.WATCH_SCENES if scenes is None else list(scenes)

        successes, steps = [], []
        for scene in scenes:
            r = self.evaluate(scenes=[scene], max_steps=max_steps, render=True)
            ok, step = r.successes[0], r.steps_to_success[0]
            print(f"scene {scene:>2}  {'ok  ' if ok else 'FAIL'}  "
                  f"({step if step is not None else max_steps} steps)",
                  flush=True)
            successes.append(ok)
            steps.append(step)

        # Printed as a count, never a percentage. At a true rate near 0.18, a
        # 5-scene sample comes up empty ~37% of the time; this is a liveness
        # read, and eval.py at n_eval=50 is the measurement. A percentage here
        # would invite quoting it as one.
        print(f"{sum(successes)}/{len(successes)}")
        return EvalResult(scenes, successes, steps)

    def _write_video(self, path, frames, fps=20):
        """MP4 of exactly what the policy saw.

        Orientation follows LIBERO's own renderer
        (benchmark_scripts/render_single_task.py:33): vertical flip, then
        RGB->BGR for cv2. The frames are the raw agentview obs, so a video that
        looks wrong here means the model's input looks wrong too.
        """
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        try:
            for f in frames:
                writer.write(f[::-1, :, ::-1])
        finally:
            writer.release()

    # ---- eval ------------------------------------------------------------
    def _rollout(self, env, batch, max_steps, render=False, zero_action=False,
                 capture=False):
        """One batch of rollouts on a fresh env. THE rollout loop.

        Every rollout in this project runs here -- scored, watched, recorded.
        Anything that needs a variation takes a flag rather than a second copy
        of the loop, because the two copies that used to exist drifted and
        disagreed about the same checkpoint.

        batch is a list of benchmark init-state indices, one per env. A batch
        of [None] means the reference protocol: two resets and no init state,
        letting the BDDL sampler pick.

        Returns (dones, done_step, frames).
        """
        n = len(batch)

        # Protocol from LIBERO's lifelong/metric.py: reset, set state, settle.
        if batch == [None]:
            env.reset()
            obs = env.reset()
        else:
            env.reset()
            obs = env.set_init_state(self._init_states()[np.array(batch)])
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(np.zeros((n, ACTION_DIM)))

        if render:
            env.render()

        dones = [False] * n
        done_step = [None] * n
        frames = [[o["agentview_image"]] for o in obs] if capture else None

        for step in range(1, max_steps + 1):
            if zero_action:
                actions = np.zeros((n, ACTION_DIM))
            else:
                actions = self._act(obs)
                assert actions.shape == (n, ACTION_DIM), (
                    f"policy returned {actions.shape}, expected {(n, ACTION_DIM)}"
                )

            obs, _, done, _ = env.step(actions)

            if render:
                env.render()

            if frames is not None:
                for k in range(n):
                    frames[k].append(obs[k]["agentview_image"])

            # Sticky: success means done fired at any point, not that it was
            # still set on the final step.
            for k in range(n):
                if not dones[k] and done[k]:
                    dones[k], done_step[k] = True, step

            if all(dones):
                break

        return dones, done_step, frames

    def evaluate(self, n_eval=50, max_steps=600, env_num=1, seed=0,
                 zero_action=False, record_dir=None, scenes=None, render=False):
        """Rollout success rate from the benchmark's fixed init states.

        scenes names the init states to run; n_eval is shorthand for the first
        n. Naming them explicitly is what lets test.py watch exactly the scenes
        eval.py scores, and lets both report per-scene outcomes that can be
        compared directly.

        zero_action ignores the model and sends zeros -- a smoke test of the
        loop that needs no trained checkpoint and must score 0%.

        record_dir writes one MP4 per rollout, named by outcome. render opens a
        live window instead (env_num=1 only). Both are parameters here rather
        than separate scripts so that watching, recording and scoring cannot
        drift apart -- see _rollout.

        A FRESH env is built for every batch and closed after it, so no rollout
        inherits simulator state from another. reset() + set_init_state() is
        NOT sufficient: the 92-dim state vector is time+qpos+qvel and does not
        carry MuJoCo's qacc_warmstart, so the contact solver warm-starts from
        whatever ran before. Measured on one checkpoint over scenes 0-19:
        reusing an env scored {4, 17}, a fresh env per rollout scored {6}, and
        the whole-run rate moved 5% -> 10% -> 15% purely with env reuse and
        env_num. Rebuilding costs env construction per batch; a score that
        depends on evaluation order costs more.
        """
        was_training = self.model.training
        self.model.eval()

        if scenes is None:
            scenes = [i % self._init_states().shape[0] for i in range(n_eval)]
        else:
            scenes = list(scenes)

        successes, steps_to_success = [], []

        for i in range(0, len(scenes), env_num):
            batch = scenes[i:i + env_num]
            # Sized to the batch, so a short final batch is not padded with
            # throwaway rollouts. Safe only because a rollout's outcome no
            # longer depends on how many envs share the step.
            env = self._build_env(len(batch), seed, render=render)
            try:
                dones, done_step, frames = self._rollout(
                    env, batch, max_steps, render=render,
                    zero_action=zero_action, capture=record_dir is not None)

                if frames is not None:
                    os.makedirs(record_dir, exist_ok=True)
                    for k, scene in enumerate(batch):
                        tag = "success" if dones[k] else "fail"
                        name = "ref" if scene is None else f"{scene:02d}"
                        path = os.path.join(
                            record_dir, f"scene{name}_{tag}.mp4")
                        self._write_video(path, frames[k])
                        print(f"  wrote {path} ({len(frames[k])} frames)",
                              flush=True)

                successes.extend(dones)
                steps_to_success.extend(done_step)
            finally:
                # Unclosed envs are the established cause of EGL shutdown
                # tracebacks in this project.
                env.close()

        if was_training:
            self.model.train()

        return EvalResult(scenes, successes, steps_to_success)

    # ---- train -----------------------------------------------------------
    def _run_tag(self):
        """Branch name for the run dir, matching the other projects' scheme.

        Prefers the remote ref pointing at HEAD so a Beekeeper run (detached
        after fetching) still names its branch; falls back to the local branch.
        """
        try:
            refs = subprocess.check_output(
                ["git", "for-each-ref", "--format=%(refname:short)",
                 "--points-at", "HEAD", "refs/remotes/origin/"],
                stderr=subprocess.DEVNULL).decode().strip()
            tag = refs.splitlines()[0].replace("origin/", "") if refs else ""
            if not tag:
                tag = subprocess.check_output(
                    ["git", "branch", "--show-current"],
                    stderr=subprocess.DEVNULL).decode().strip()
            return tag or "unknown"
        except Exception:
            return "unknown"

    def train(self, steps, batch_size, eval_every=None, n_eval=10,
              eval_max_steps=300, eval_env_num=10, log_dir="runs",
              run_tag=None, save_every=10000):
        # Timestamped run dir: TensorBoard's whole point is overlaying runs, so
        # they must not overwrite each other. Name matches the convention in
        # sac-homebot-route-planner / q-homebot-route-planner so one
        # `tensorboard --logdir` habit works across projects.
        if run_tag is None:
            run_tag = self._run_tag()
        run_dir = os.path.join(
            log_dir, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')
        writer = SummaryWriter(run_dir)
        print(f"logging to {run_dir}  ->  tensorboard --logdir {log_dir}",
              flush=True)

        # "steps", not epochs: get_batch samples with replacement, so there are
        # no epoch boundaries. 100k steps at batch 32 is ~632 effective passes
        # over the 5068-transition dataset.
        for i in range(steps):
            batch = self.dl.get_batch(batch_size=batch_size)

            images = batch["agentview"].to(self.device)
            joint_states = batch["joint_state"].to(self.device)
            actions = batch["actions"].to(self.device)

            actions_pred = self.model(images, joint_states)
            loss = F.l1_loss(actions_pred, actions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"step {i} loss {loss.item():.4f}")
                writer.add_scalar("train/l1_loss", loss.item(), i)

            # Periodic, unconditional save -- the checkpoint on disk is always
            # the LATEST weights, not the lowest-loss ones. Run 427 measured why
            # that matters: loss kept falling for 50k steps after success rate
            # went flat, so best-loss selection was picking a model that scored
            # worse than earlier ones it had already discarded.
            if save_every and i > 0 and i % save_every == 0:
                self.model.save_checkpoint()
                print(f"step {i} saved {self.model.checkpoint_file}", flush=True)

            # Same code path as standalone eval, so the two cannot drift.
            # eval_max_steps is shorter than the standalone default: median
            # success is ~108 steps, so 300 only truncates rollouts that were
            # going to fail, and halves the cost of the early evals where
            # nothing succeeds and every rollout runs to the cap.
            if eval_every and i > 0 and i % eval_every == 0:
                self._log_eval(writer, i, n_eval, eval_max_steps, eval_env_num)

        # Save before the final eval, not after: eval builds envs and can fail,
        # and losing the finished weights to a rendering error would be absurd.
        self.model.save_checkpoint()
        print(f"step {steps} saved {self.model.checkpoint_file}", flush=True)

        # range(steps) stops at steps-1, so the loop above never evaluates the
        # final model. Score it explicitly rather than finishing untested.
        if eval_every:
            self._log_eval(writer, steps, n_eval, eval_max_steps, eval_env_num)

        writer.close()

    def _log_eval(self, writer, step, n_eval, max_steps, env_num):
        result = self.evaluate(n_eval=n_eval, max_steps=max_steps,
                               env_num=env_num)
        print(f"step {step} {result}", flush=True)
        writer.add_scalar("eval/success_rate", result.success_rate, step)
        solved = [s for s in result.steps_to_success if s is not None]
        if solved:
            writer.add_scalar("eval/median_steps", float(np.median(solved)), step)
        writer.flush()
        return result
