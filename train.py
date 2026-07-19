import os
import sys
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

# import libero.libero.envs.env_wrapper as ew
# import libero.libero.envs as lenv
#
# print([e for e in dir(lenv) if e.endswith("Env")])
# print(lenv.__file__)
# # print([w for w in dir(ew) if w.endswith("Env")])
# print(ew.__file__)

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)
task_bddl = os.path.join(get_libero_path("bddl_files"),
                         task.problem_folder, task.bddl_file)

env = ControlEnv(bddl_file_name=task_bddl,
                 camera_heights=128,
                 camera_widths=128,
                 has_renderer=True,
                 use_camera_obs=True,
                 render_camera="agentview")
# env = OffScreenRenderEnv(bddl_file_name=task_bddl,
                         # camera_heights=128, camera_widths=128)
env.seed(0)
env.reset()

init_states = task_suite.get_task_init_states(0)
obs = env.set_init_state(init_states[0])

for i in range(100):
    action = np.random.uniform(-1, 1, 7)

    obs, reward, done, info = env.step(action)  # action: 7-dim
    env.env.render()

#print(obs["agentview_image"].shape, obs["agentview_image"].mean())
env.close()
