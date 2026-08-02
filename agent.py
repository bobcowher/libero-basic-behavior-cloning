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

IMG_SIZE = 128
ACTION_DIM = 7

# Steps of zero action after set_init_state, before the policy sees anything.
# Copied from LIBERO's own eval loop (lifelong/metric.py). Without it the first
# observation shows a mid-transient arm that matches no training frame.
SETTLE_STEPS = 5


@dataclass
class EvalResult:
    success_rate: float
    successes: list          # per-rollout bool
    steps_to_success: list   # step index of success, None on failure

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
        images, proprio = self._preprocess_obs(raw_obs)
        return self.model(images, proprio).cpu().numpy()

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
    def _build_env(self, env_num, seed):
        bddl = os.path.join(get_libero_path("bddl_files"),
                            self.task.problem_folder, self.task.bddl_file)
        env_args = {"bddl_file_name": bddl,
                    "camera_heights": IMG_SIZE, "camera_widths": IMG_SIZE}

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
                env = cls(
                    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                )
                env.seed(seed)
                return env
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5)
        raise RuntimeError("unreachable: env creation loop exhausted")

    # ---- watch ------------------------------------------------------------
    def test(self, scene=0, max_steps=1000):
        """Run the policy in a live viewer window.

        Both renderers on at once: has_renderer draws the window,
        has_offscreen_renderer + use_camera_obs give the policy its agentview
        images. Watching and acting are not mutually exclusive.

        Needs MUJOCO_GL=glfw and a display -- test.py sets that before this
        module is imported, since mujoco picks its backend at import time.

        Starts from the same benchmark init state eval scores (`scene` indexes
        the 50), and takes the same settle steps, so what you watch is a
        rollout eval.py would have counted. Actions come from self._act, the
        one preprocessing path.
        """
        self.model.eval()

        bddl = os.path.join(get_libero_path("bddl_files"),
                            self.task.problem_folder, self.task.bddl_file)
        env = ControlEnv(bddl_file_name=bddl,
                         has_renderer=True,
                         has_offscreen_renderer=True,
                         use_camera_obs=True,
                         render_camera="agentview",
                         camera_heights=IMG_SIZE, camera_widths=IMG_SIZE)
        env.seed(0)

        try:
            env.reset()
            obs = env.set_init_state(self._init_states()[scene])
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(np.zeros(ACTION_DIM))

            for step in range(1, max_steps + 1):
                obs, _, done, _ = env.step(self._act([obs])[0])
                # ControlEnv has no render(); the window belongs to the
                # robosuite env it wraps. LiberoObsWrapper did the same.
                env.env.render()
                if done:
                    print(f"success at step {step}")
                    return True
            print(f"no success in {max_steps} steps")
            return False
        finally:
            env.close()

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
    def evaluate(self, n_eval=50, max_steps=600, env_num=1, seed=0,
                 zero_action=False, record_dir=None):
        """Rollout success rate from the benchmark's fixed init states.

        Protocol mirrors LIBERO's lifelong/metric.py. zero_action ignores the
        model and sends zeros -- a smoke test of the loop that needs no trained
        checkpoint and must score 0%.

        record_dir writes one MP4 per rollout, named by outcome. Recording is a
        parameter here rather than a separate script so that watching a rollout
        and scoring one cannot drift apart -- there is one rollout loop.
        """
        was_training = self.model.training
        self.model.eval()

        env = self._build_env(env_num, seed)
        init_states = self._init_states()

        successes, steps_to_success = [], []
        eval_loop_num = (n_eval + env_num - 1) // env_num

        try:
            for i in range(eval_loop_num):
                env.reset()
                idx = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
                obs = env.set_init_state(init_states[idx])

                for _ in range(SETTLE_STEPS):
                    obs, _, _, _ = env.step(np.zeros((env_num, ACTION_DIM)))

                dones = [False] * env_num
                done_step = [None] * env_num
                frames = [[o["agentview_image"]] for o in obs] if record_dir else None

                for step in range(1, max_steps + 1):
                    if zero_action:
                        actions = np.zeros((env_num, ACTION_DIM))
                    else:
                        actions = self._act(obs)
                        assert actions.shape == (env_num, ACTION_DIM), (
                            f"policy returned {actions.shape}, "
                            f"expected {(env_num, ACTION_DIM)}"
                        )

                    obs, _, done, _ = env.step(actions)

                    if frames is not None:
                        for k in range(env_num):
                            frames[k].append(obs[k]["agentview_image"])

                    # Sticky: success means done fired at any point, not that
                    # it was still set on the final step.
                    for k in range(env_num):
                        if not dones[k] and done[k]:
                            dones[k], done_step[k] = True, step

                    if all(dones):
                        break

                if frames is not None:
                    os.makedirs(record_dir, exist_ok=True)
                    for k in range(env_num):
                        tag = "success" if dones[k] else "fail"
                        path = os.path.join(
                            record_dir, f"rollout{i * env_num + k:02d}_{tag}.mp4")
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

        successes = successes[:n_eval]
        steps_to_success = steps_to_success[:n_eval]
        return EvalResult(sum(successes) / len(successes), successes, steps_to_success)

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
              run_tag=None):
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

        lowest_loss = float("inf")

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
                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                    self.model.save_checkpoint()

            # Same code path as standalone eval, so the two cannot drift.
            # eval_max_steps is shorter than the standalone default: median
            # success is ~108 steps, so 300 only truncates rollouts that were
            # going to fail, and halves the cost of the early evals where
            # nothing succeeds and every rollout runs to the cap.
            if eval_every and i > 0 and i % eval_every == 0:
                self._log_eval(writer, i, n_eval, eval_max_steps, eval_env_num)

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
