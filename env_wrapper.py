import numpy as np


class LiberoObsWrapper:
    """Gym-style wrapper over a LIBERO env (OffScreenRenderEnv / ControlEnv).

    Collapses the raw obs dict down to exactly what the policy consumes:
      - image   (3, H, W) float32 in [0, 1], CHW      <- one camera
      - joint_state (9,)       float32, RAW (unnormalized) <- joint_pos + gripper_qpos

    Everything else in the obs dict is dropped.

    Not a literal gym.Wrapper subclass: LIBERO's ControlEnv is not a gym.Env
    (no observation_space/action_space), so subclassing gym.Wrapper would trip
    its space assertions. This exposes the same reset()/step() surface instead.

    Train/eval matching (see CONTEXT.md):
      - live keys differ from HDF5: agentview_image / robot0_joint_pos /
        robot0_gripper_qpos  (vs agentview_rgb / joint_states / gripper_states).
      - image: HWC uint8 [0,255] -> CHW float32 [0,1]. NO vertical flip; the
        stored demos and the live env share the opengl convention.
      - joint_state is raw here because the training loader delivers it raw too. If
        you normalize joint_state for training, normalize it here with the SAME
        per-dim stats or you reintroduce skew.
    """

    def __init__(self, env, image_key="agentview_image"):
        self.env = env
        self.image_key = image_key

    def _process(self, obs):
        image = obs[self.image_key]                       # (H, W, 3) uint8
        image = np.transpose(image, (2, 0, 1))            # -> (3, H, W)
        image = image.astype(np.float32) / 255.0          # -> [0, 1]

        joint_state = np.concatenate([
            obs["robot0_joint_pos"],                      # (7,) joints
            obs["robot0_gripper_qpos"],                   # (2,) gripper
        ]).astype(np.float32)                             # -> (9,)

        return image, joint_state

    def render(self):
        self.env.env.render()

    def reset(self):
        # LIBERO reset() returns a single obs dict (not a gym (obs, info) tuple).
        obs = self.env.reset()
        return self._process(obs)

    def set_init_state(self, init_state):
        # Rollouts start from the demo's fixed init state, not a random reset.
        obs = self.env.set_init_state(init_state)
        return self._process(obs)

    def step(self, action):
        # robosuite returns a 4-tuple (obs, reward, done, info), not gym's 5.
        obs, reward, done, info = self.env.step(action)
        image, joint_state = self._process(obs)
        return (image, joint_state), reward, done, info

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        return self.env.close()

