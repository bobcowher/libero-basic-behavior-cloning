import contextlib
import io
import os
import sys
import numpy as np
from robosuite.controllers import joint_pos
import torch
import h5py
import inspect

from torch._C import device
from dataloader import DataLoader
from env_wrapper import LiberoObsWrapper
from model import Model
import torch.nn as nn
import torch.nn.functional as F

# libero pulls in `gym`, which prints an unmaintained-package notice straight
# to stderr on import (gym_notices) instead of raising a warnings.UserWarning,
# so PYTHONWARNINGS can't filter it. Swallow stderr just for this import.
with contextlib.redirect_stderr(io.StringIO()):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

class Agent:

    def __init__(self, eval=False, lr=0.0001):

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

        # Wrap: raw obs dict -> (image (3,128,128) f32 [0,1], joint_state (9,) f32).
        self.env = LiberoObsWrapper(self.env)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Image-input-only policy. input_shape is the wrapped image shape;
        # joint_state is NOT fed in this V1 (model.forward ignores joint_state).
        self.model = Model(input_shape=(3, 128, 128), num_actions=7, hidden_dim=256).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        image, joint_state = self.env.reset()               # (image, joint_state)
        print("image", image.shape, image.dtype, "joint_state", joint_state.shape)
    
    def train(self, epochs, batch_size):

        lowest_loss = 100 # Arbitrarily high number

        for i in range(epochs):

            batch = self.dl.get_batch(batch_size=batch_size)

            images = batch['agentview'].to(self.device)
            joint_states = batch['joint_state'].to(self.device)
            actions = batch['actions'].to(self.device)

            # TODO: Go integrate joint states
            actions_pred = self.model(images, joint_states)

            loss = F.l1_loss(actions, actions_pred)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Episode: {i} Loss: {loss.item()}")
                
                if(loss.item() < lowest_loss):
                    lowest_loss = loss.item()
                    self.model.save_checkpoint()
                    print(f"\nSaved checkpoint at episode {i}\n")

    def test(self):
        self.model.load_checkpoint()

        image, joint_state = self.env.reset()

        for i in range(3000):

            action = self.model(image, joint_state)
            obs, reward, done, info = self.env.step(action)
            image, joint_state = obs.image.to(self.device), obs.joint_state.to(self.device)



        


    def close(self):
        self.env.close()

        # print([k for k in obs.keys() if "image" in k])
        #print(obs["agentview_image"].shape, obs["agentview_image"].mean())
        # env.close()

