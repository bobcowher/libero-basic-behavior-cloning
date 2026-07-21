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
        # device: where get_batch places tensors ("cpu" or e.g. "cuda:0").
        #   Stored as the default; get_batch(device=...) can still override.
        # seq_len: length of each obs/action window robomimic returns. The V1
        #   policy is single-step (uses index [0] only), so seq_len=1 is correct
        #   and avoids caching frames you never read. seq_len applies to obs AND
        #   actions (robomimic is built for BC-RNN); the eventual ACT plan needs
        #   a 16-length ACTION window but still only 1 image -- robomimic can't
        #   express that asymmetry, so revisit then.
        # hdf5_cache_mode: robomimic cache strategy.
        #   "low_dim" caches only proprio in RAM; images are read from disk on
        #   every dataset[i]. "all" caches the fully-processed get_item results
        #   in RAM -- MEASURED ~2GB at seq_len=1, but ~30GB at seq_len=16 (it
        #   caches a full float32 window per sample). Only pair "all" with a
        #   small seq_len. Host RAM only; never GPU.
        self.device = device

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
            seq_len=seq_len,
            frame_stack=1,
            hdf5_cache_mode=hdf5_cache_mode,
        )

        self.dataset = dataset
        self.shape_meta = shape_meta

    def get_batch(self, batch_size=64, device=None):
        # device=None -> use the DataLoader's configured self.device.
        device = self.device if device is None else device
        # Random sample WITH replacement (replay-buffer idiom, no epoch
        # boundaries). Single-step pairs: obs at index 0 with the action taken
        # at that same step, actions[0]. NOTE: assumes obs[0] and actions[0]
        # share a timestep — the replay gate must confirm before you trust it.
        #
        # Index [0] of each window is used. At seq_len=1 that's the whole
        # window (no waste). At larger seq_len the extra frames are cached but
        # discarded here -- see the seq_len note in __init__.
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
