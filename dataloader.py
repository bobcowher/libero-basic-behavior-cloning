import numpy as np
import torch
import contextlib
import io
import os

with contextlib.redirect_stderr(io.StringIO()):
    import libero.lifelong.datasets as D
    from libero.libero import benchmark, get_libero_path

class DataLoader():
    def __init__(self, dataset_filename, device="cpu", seq_len=1,
                 hdf5_cache_mode="low_dim"):
        # seq_len applies to obs AND actions (robomimic is built for BC-RNN).
        #   The eventual ACT plan needs a 16-length action window but only 1
        #   image; robomimic can't express that asymmetry, so revisit then.
        # hdf5_cache_mode="all" caches processed windows in host RAM -- measured
        #   ~2GB at seq_len=1 but ~30GB at seq_len=16. Only pair it with a small
        #   seq_len. "low_dim" caches proprio only and reads images from disk.
        self.device = device

        # Relative to the datasets root, including the problem folder:
        # "libero_spatial/<task>_demo.hdf5".
        dataset_file = os.path.join(
            get_libero_path("datasets"),
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
            seq_len=seq_len,
            frame_stack=1,
            hdf5_cache_mode=hdf5_cache_mode,
        )

        self.dataset = dataset
        self.shape_meta = shape_meta

    def get_batch(self, batch_size=64, device=None):
        # device=None -> use the DataLoader's configured self.device.
        device = self.device if device is None else device
        # Sampled with replacement -- replay-buffer idiom, no epoch boundaries.
        # Index [0] of each window: obs paired with the action at that same
        # step. At seq_len > 1 the extra frames are cached but discarded here.
        idxs = np.random.randint(0, len(self.dataset), size=batch_size)

        agentview, wrist, joint_state, actions = [], [], [], []
        for i in idxs:
            d = self.dataset[i]
            agentview.append(d["obs"]["agentview_rgb"][0])
            wrist.append(d["obs"]["eye_in_hand_rgb"][0])
            joint_state.append(np.concatenate([
                d["obs"]["joint_states"][0],
                d["obs"]["gripper_states"][0],
            ]))
            actions.append(d["actions"][0])

        def stack(arr):
            return torch.as_tensor(np.stack(arr)).float().to(device)

        return {
            "agentview": stack(agentview),   # (B, 3, 128, 128)
            "wrist":     stack(wrist),        # (B, 3, 128, 128)
            "joint_state":   stack(joint_state),      # (B, 9)
            "actions":   stack(actions),      # (B, 7)
        }
