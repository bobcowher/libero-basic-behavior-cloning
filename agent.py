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

# cuDNN TF32 is on by default on Ampere+ and makes conv output batch-size
# dependent: max action delta batch-1 vs batch-10 is 1.64e-4 with it, 1.49e-8
# without. Closed-loop rollouts amplify that into different solved scenes.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True



class WatchEnv(ControlEnv):
    """OffScreenRenderEnv with the on-screen window enabled.

    A separate class only because OffScreenRenderEnv hardcodes
    has_renderer=False (env_wrapper.py:152). Watching and scoring must differ
    by this boolean and nothing else.
    """

    def __init__(self, **kwargs):
        kwargs["has_renderer"] = True
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)

    def render(self, **kwargs):
        # ControlEnv defines no render(); the window belongs to the wrapped env.
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

        # What we configure the env's cameras to -- an input, not a derived value.
        self.img_size = 128

        # Zero-action steps after set_init_state, from LIBERO's lifelong/metric.py.
        # Without them the first obs is a mid-transient arm matching no training frame.
        self.settle_steps = 5

        # The camera the policy sees. Wrist-only: a real arm has no third-person
        # view. The live env and the HDF5 loader name the same camera
        # differently, so both keys live here -- swapping cameras must move
        # train and eval together.
        self.policy_cam_env = "robot0_eye_in_hand_image"   # live obs dict
        self.policy_cam_batch = "wrist"                    # DataLoader.get_batch

        # Third-person, for recorded video and the watch window only. Never fed
        # to the policy.
        self.human_cam_env = "agentview_image"

        # Must follow the camera and size settings -- it builds an env.
        self.action_dim, self.proprio_dim, image_shape = self._probe_dims()

        # --- Architecture knobs (parametric sweep) -----------------------
        # Each experiment branch edits ONLY these three numbers. Baseline is
        # hidden_dim=256, compression_dim=256 (tied), n_hidden_layers=1.
        self.hidden_dim = 256
        self.compression_dim = 256
        self.n_hidden_layers = 1
        # -----------------------------------------------------------------

        self.model = Model(
            image_input_shape=image_shape,
            joint_input_dim=self.proprio_dim,
            num_actions=self.action_dim,
            hidden_dim=self.hidden_dim,
            compression_dim=self.compression_dim,
            n_hidden_layers=self.n_hidden_layers,
            checkpoint_dir=os.path.dirname(ckpt) or ".",
            name=os.path.basename(ckpt),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._dl = None

    # ---- shapes ----------------------------------------------------------
    @staticmethod
    def _proprio(obs):
        """The proprio vector. Used to size the model AND to feed it, so the
        two cannot drift."""
        return np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])

    def _probe_dims(self):
        """(action_dim, proprio_dim, image_shape) from a throwaway env, ~2.8s.

        Safe to reset: the BDDL placement sampler is per-instance, so this does
        not shift the scenes a later env samples (verified, distance exactly 0).
        """
        env = self._build_env(1, seed=0)
        try:
            # NOT robots[0].dof -- that counts the gripper as 1 actuated DOF
            # while the obs reports 2 finger positions.
            robot = env.get_env_attr("robots")[0][0]   # worker 0, arm 0
            action_dim = robot.action_dim

            obs = env.reset()[0]
            proprio_dim = self._proprio(obs).shape[0]
            h, w, c = obs[self.policy_cam_env].shape        # -> (C,H,W) for the model
            return action_dim, proprio_dim, (c, h, w)
        finally:
            env.close()

    # ---- lazy dataset ----------------------------------------------------
    # Lazy so scoring a checkpoint never builds the training set.
    @property
    def dl(self):
        if self._dl is None:
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
        """Live LIBERO obs dicts -> (images, proprio) device tensors.

        The only live-env preprocessing path. HDF5 and live envs use different
        keys for the same quantities (eye_in_hand_rgb vs robot0_eye_in_hand_image,
        joint_states vs robot0_joint_pos) -- the top source of silent
        train/eval skew, so the translation exists once.

        No vertical flip: demos and the live env share the opengl convention.
        Proprio is raw because the loader delivers it raw; if training ever
        normalizes, normalize here with the same stats.
        """
        images = np.stack([o[self.policy_cam_env] for o in raw_obs])      # (N,H,W,3) uint8
        images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        proprio = np.stack([self._proprio(o) for o in raw_obs]
                           ).astype(np.float32)                      # (N, 9)

        return (torch.from_numpy(images).to(self.device),
                torch.from_numpy(proprio).to(self.device))

    @torch.no_grad()
    def _act(self, raw_obs):
        """Actions for a list of live obs dicts.

        One observation at a time, even with several envs in flight: batched
        conv reductions are batch-size dependent (~1e-8 even without TF32), and
        that was enough to change which scenes evaluate() solved. The MuJoCo
        step dominates wall clock, so this costs nothing that matters.
        """
        images, proprio = self._preprocess_obs(raw_obs)
        return np.stack([
            self.model(images[i:i + 1], proprio[i:i + 1])[0].cpu().numpy()
            for i in range(images.shape[0])
        ])

    def _init_states(self):
        """The benchmark's 50 fixed eval init states.

        Replicates benchmark.get_task_init_states (benchmark/__init__.py:158)
        rather than calling it: that helper omits weights_only, which torch
        >= 2.6 defaults to True, rejecting the numpy pickle. Safe here -- local
        benchmark data, not a downloaded checkpoint.
        """
        path = os.path.join(get_libero_path("init_states"),
                            self.task.problem_folder, self.task.init_states_file)
        return torch.load(path, weights_only=False)

    # ---- env -------------------------------------------------------------
    def _build_env(self, env_num, seed, render=False):
        bddl = os.path.join(get_libero_path("bddl_files"),
                            self.task.problem_folder, self.task.bddl_file)
        env_args = {"bddl_file_name": bddl,
                    "camera_heights": self.img_size, "camera_widths": self.img_size}

        if render:
            assert env_num == 1, "render requires env_num=1"
            # Viewer camera only -- does not touch camera_names, so the policy's
            # agentview_image is unaffected.
            env_args["render_camera"] = "agentview"
            make = lambda: WatchEnv(**env_args)          # noqa: E731
        else:
            make = lambda: OffScreenRenderEnv(**env_args)  # noqa: E731

        cls = DummyVectorEnv if env_num == 1 else SubprocVectorEnv

        if env_num > 1:
            # Spawn, not fork: CUDA is already initialized in the parent and
            # driver state does not survive fork() -- forked workers die with
            # EGL_BAD_ALLOC. Requires entry points to guard __main__.
            mp.set_start_method("spawn", force=True)

        # Env creation intermittently fails on a frame buffer error; LIBERO's
        # own eval retries too (lifelong/metric.py:85-100).
        for attempt in range(5):
            try:
                env = cls([make for _ in range(env_num)])
                # A list, not an int: seed(int) expands to seed+i per worker
                # (venv.py:849), making a scene's result depend on env_num.
                env.seed([seed] * env_num)
                return env
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5)
        raise RuntimeError("unreachable: env creation loop exhausted")

    # ---- watch ------------------------------------------------------------
    # A stride across the whole 50-scene set; the first N would be a corner of
    # it.
    WATCH_SCENES = list(range(0, 50, 10))

    def test(self, scenes=None, max_steps=300, reference_protocol=False):
        """Watch the policy run, in a window, on the scenes eval.py scores.

        A view of the eval, not a separate thing -- every rollout goes through
        evaluate(), so a scene seen succeeding here is one counted there.

        Needs MUJOCO_GL=glfw and a display; test.py sets that before importing
        this module, since mujoco picks its backend at import.

        One scene per evaluate() call so outcomes print as they happen. Each
        rollout still gets a fresh env, so the window reopens between scenes.

        max_steps=300 vs eval.py's 600: successes finish near ~101 steps, so
        only doomed rollouts are truncated.

        reference_protocol reproduces the reference project's setup (seed 0,
        two resets, no init state, no settle). NOT a scored scene -- reset()
        samples fresh placements, a third distribution.
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

        # A count, never a percentage: at a true rate near 0.18 a 5-scene sample
        # is empty ~37% of the time. This is a liveness read, eval.py measures.
        print(f"{sum(successes)}/{len(successes)}")
        return EvalResult(scenes, successes, steps)

    def _write_video(self, path, frames, fps=20):
        """MP4 of exactly what the policy saw -- a video that looks wrong means
        the model's input is wrong.

        Vertical flip then RGB->BGR, following LIBERO's own renderer
        (benchmark_scripts/render_single_task.py:33).
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

        Every rollout runs here -- scored, watched, recorded. Variations are
        flags, not copies; the two copies that used to exist drifted and
        disagreed about the same checkpoint.

        batch is a list of init-state indices, one per env; [None] means the
        reference protocol. Returns (dones, done_step, frames).
        """
        n = len(batch)

        # Protocol from LIBERO's lifelong/metric.py: reset, set state, settle.
        if batch == [None]:
            env.reset()
            obs = env.reset()
        else:
            env.reset()
            obs = env.set_init_state(self._init_states()[np.array(batch)])
            for _ in range(self.settle_steps):
                obs, _, _, _ = env.step(np.zeros((n, self.action_dim)))

        if render:
            env.render()

        dones = [False] * n
        done_step = [None] * n
        # Third-person for the video even though the policy sees the wrist --
        # a wrist recording can't show whether the task actually got solved.
        frames = [[o[self.human_cam_env]] for o in obs] if capture else None

        for step in range(1, max_steps + 1):
            if zero_action:
                actions = np.zeros((n, self.action_dim))
            else:
                actions = self._act(obs)
                assert actions.shape == (n, self.action_dim), (
                    f"policy returned {actions.shape}, "
                    f"expected {(n, self.action_dim)}"
                )

            obs, _, done, _ = env.step(actions)

            if render:
                env.render()

            if frames is not None:
                for k in range(n):
                    frames[k].append(obs[k][self.human_cam_env])

            # Sticky: done fired at any point counts, not just on the last step.
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
        n. zero_action sends zeros instead of model output -- a smoke test that
        must score 0%. record_dir writes one MP4 per rollout; render opens a
        live window (env_num=1 only).

        A fresh env per batch, closed after: reset() + set_init_state() is NOT
        a clean slate, because the 92-dim state is time+qpos+qvel and omits
        qacc_warmstart, so the contact solver warm-starts from the previous
        rollout. Measured over scenes 0-19: a reused env solved {4,17}, a fresh
        one solved {6}.
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
            # Sized to the batch so a short final one is not padded. Safe only
            # because outcomes no longer depend on how many envs share a step.
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
                # Unclosed envs cause EGL shutdown tracebacks.
                env.close()

        if was_training:
            self.model.train()

        return EvalResult(scenes, successes, steps_to_success)

    # ---- train -----------------------------------------------------------
    def _run_tag(self):
        """Branch name for the run dir.

        Prefers the remote ref at HEAD so a Beekeeper run (detached after
        fetching) still names its branch; falls back to the local branch.
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
        # Timestamped run dir so runs overlay instead of overwriting. Naming
        # matches sac-/q-homebot-route-planner.
        if run_tag is None:
            run_tag = self._run_tag()
        # Encode the swept architecture in the run name so TensorBoard shows
        # what was tested at a glance (e.g. ..._exp-hdim-128_h128_c128_L1).
        arch = f"h{self.hidden_dim}_c{self.compression_dim}_L{self.n_hidden_layers}"
        run_tag = f"{run_tag}_{arch}"
        run_dir = os.path.join(
            log_dir, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')
        writer = SummaryWriter(run_dir)
        writer.add_text(
            "config/arch",
            f"hidden_dim={self.hidden_dim}  compression_dim={self.compression_dim}"
            f"  n_hidden_layers={self.n_hidden_layers}", 0)
        print(f"logging to {run_dir}  ->  tensorboard --logdir {log_dir}",
              flush=True)

        # Steps, not epochs: get_batch samples with replacement, so there are no
        # epoch boundaries. 100k steps at batch 32 is ~632 passes over 5068.
        for i in range(steps):
            batch = self.dl.get_batch(batch_size=batch_size)

            images = batch[self.policy_cam_batch].to(self.device)
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

            # Latest weights, not lowest-loss: run 427 saw loss fall for 50k
            # steps after success rate went flat, so best-loss selection was
            # discarding better policies.
            if save_every and i > 0 and i % save_every == 0:
                self.model.save_checkpoint()
                print(f"step {i} saved {self.model.checkpoint_file}", flush=True)

            # eval_max_steps=300 vs the standalone 600: median success is ~108
            # steps, so this only truncates rollouts that were going to fail.
            if eval_every and i > 0 and i % eval_every == 0:
                self._log_eval(writer, i, n_eval, eval_max_steps, eval_env_num)

        # Before the final eval, not after -- eval builds envs and can fail.
        self.model.save_checkpoint()
        print(f"step {steps} saved {self.model.checkpoint_file}", flush=True)

        # range(steps) stops at steps-1, so the loop never scores the final
        # model.
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
