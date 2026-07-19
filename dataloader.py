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
            framee_stack=1,
            hdf5_cache_mode="low_dim",
        )

        self.dataset = dataset
        self.shape_meta = shape_meta


class SingleObsChunk(torch.utils.data.Dataset):
    def __init__(self, seq_ds):
        self.ds = seq_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        d = self.ds[i]
        return {
            "agentview": d["obs"]["agentview_rgb"][0],    # (3,128,128)
            "wrist":     d["obs"]["eye_in_hand_rgb"][0],
            "proprio":   np.concatenate([d["obs"]["joint_states"][0],
                                         d["obs"]["gripper_states"][0]]),
            "actions":   d["actions"],                    # (16,7)
        }
