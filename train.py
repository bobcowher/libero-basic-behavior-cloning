import contextlib
import io
import os
import sys
import numpy as np

# libero pulls in `gym`, which prints an unmaintained-package notice straight
# to stderr on import (gym_notices) instead of raising a warnings.UserWarning,
# so PYTHONWARNINGS can't filter it. Swallow stderr just for this import.
with contextlib.redirect_stderr(io.StringIO()):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

print(get_libero_path("datasets"))

sys.exit()

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)
task_bddl = os.path.join(get_libero_path("bddl_files"),
                         task.problem_folder, task.bddl_file)
view = True 

if view:
    env = ControlEnv(bddl_file_name=task_bddl, 
                     has_renderer=True,
                     has_offscreen_renderer=True, 
                     use_camera_obs=True,
                     render_camera="agentview")
else:
    env = OffScreenRenderEnv(bddl_file_name=task_bddl,
                             camera_heights=128, camera_widths=128)

env.seed(0)
env.reset()

init_states = task_suite.get_task_init_states(0)
obs = env.set_init_state(init_states[0])

# Close out early. 

for i in range(100):
    action = np.concatenate((
        np.random.uniform(-0.3, 0.3, 6), [np.random.choice([-1, 1])]
    ))
    # action = np.random.uniform(-0.3, 0.03, 6)

    obs, reward, done, info = env.step(action)  # action: 7-dim

    if view:
        env.env.render()

    if done:
        break

# print([k for k in obs.keys() if "image" in k])
#print(obs["agentview_image"].shape, obs["agentview_image"].mean())
env.close()
