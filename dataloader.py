import numpy as np
import torch
import contextlib
import io
import os

with contextlib.redirect_stderr(io.StringIO()):
    import libero.lifelong.datasets as D
    from libero.libero import benchmark, get_libero_path

class DataLoader():
    def __init__(self, dataset_filename):

        dataset_file = os.path.join(
            get_libero_path("datasets"),
            "libero_spatial",
            dataset_filename
        )

        obs_modality = {
            "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
            "low_dim": ["joint_states", "gripper_states"],
        }

        dataset, shape_meta = D.get_dataset(
            dataset_path=dataset_file,
            obs_modality=obs_modality,
            initialize_obs_utils=True,
            seq_len=16,
            frame_stack=1,
            hdf5_cache_mode="low_dim",
        )

        self.dataset = dataset
        self.shape_meta = shape_meta

    def get_batch(self, batch_size=64, device="cpu"):
        # Random sample WITH replacement (replay-buffer idiom, no epoch
        # boundaries). Single-step pairs: obs at index 0 with the action taken
        # at that same step, actions[0]. NOTE: assumes obs[0] and actions[0]
        # share a timestep — the replay gate must confirm before you trust it.
        #
        # seq_len=16 windows are still pulled but only index 0 is used, so each
        # sample reads 16 frames to keep 1. Harmless at this scale;
        # hdf5_cache_mode="all" removes the waste if it ever bottlenecks.
        idxs = np.random.randint(0, len(self.dataset), size=batch_size)

        agentview, wrist, proprio, actions = [], [], [], []
        for i in idxs:
            d = self.dataset[i]
            agentview.append(d["obs"]["agentview_rgb"][0])
            wrist.append(d["obs"]["eye_in_hand_rgb"][0])
            proprio.append(np.concatenate([
                d["obs"]["joint_states"][0],
                d["obs"]["gripper_states"][0],
            ]))
            actions.append(d["actions"][0])

        def stack(arr):
            return torch.as_tensor(np.stack(arr)).float().to(device)

        return {
            "agentview": stack(agentview),   # (B, 3, 128, 128)
            "wrist":     stack(wrist),        # (B, 3, 128, 128)
            "proprio":   stack(proprio),      # (B, 9)
            "actions":   stack(actions),      # (B, 7)
        }
