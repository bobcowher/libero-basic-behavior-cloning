import contextlib
import io
import os
import sys
import numpy as np
import torch
import h5py
import inspect
from dataloader import DataLoader
from env_wrapper import LiberoObsWrapper
from model import Model

# libero pulls in `gym`, which prints an unmaintained-package notice straight
# to stderr on import (gym_notices) instead of raising a warnings.UserWarning,
# so PYTHONWARNINGS can't filter it. Swallow stderr just for this import.
with contextlib.redirect_stderr(io.StringIO()):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

class Agent:

    def __init__(self, eval=False):

        dataset_filename = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
        self.dl = DataLoader(dataset_filename=dataset_filename)


        task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = task_suite.get_task(0)
        task_bddl = os.path.join(get_libero_path("bddl_files"),
                                 task.problem_folder, task.bddl_file)
        self.eval = eval

        if self.eval:
            self.env = ControlEnv(bddl_file_name=task_bddl, 
                             has_renderer=True,
                             has_offscreen_renderer=True, 
                             use_camera_obs=True,
                             render_camera="agentview")
        else:
            self.env = OffScreenRenderEnv(bddl_file_name=task_bddl,
                                     camera_heights=128, camera_widths=128)
        
        self.env.seed(0)

        # Wrap: raw obs dict -> (image (3,128,128) f32 [0,1], proprio (9,) f32).
        self.env = LiberoObsWrapper(self.env)

        # Image-input-only policy. input_shape is the wrapped image shape;
        # proprio is NOT fed in this V1 (model.forward ignores joint_state).
        self.model = Model(input_shape=(3, 128, 128), num_actions=7, hidden_dim=256)

        self.obs = self.env.reset()               # (image, proprio)
        image, proprio = self.obs
        print("image", image.shape, image.dtype, "proprio", proprio.shape)
    
    def train(self, epochs, batch_size):

        for i in range(100):
            image, proprio = self.obs

            # Image input only. Batch dim added; joint_state passed as None
            # since the model ignores it in this V1.
            image_t = torch.as_tensor(image).unsqueeze(0)   # (1, 3, 128, 128)
            features = self.model(image_t, None)            # (1, hidden_dim)

            # NOTE: model has no action head yet, so `features` is not a 7-dim
            # action. Until model.py adds an output layer, step a placeholder.
            action = np.concatenate((
                np.random.uniform(-0.3, 0.3, 6), [np.random.choice([-1, 1])]
            ))

            self.obs, reward, done, info = self.env.step(action)  # action: 7-dim

            if done:
                break

    def close(self):
        self.env.close()

        # print([k for k in obs.keys() if "image" in k])
        #print(obs["agentview_image"].shape, obs["agentview_image"].mean())
        # env.close()

