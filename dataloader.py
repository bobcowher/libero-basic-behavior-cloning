import numpy as np
import torch
import contextlib
import io
import os

with contextlib.redirect_stderr(io.StringIO()):
    import libero.lifelong.datasets as D
    from libero.libero import benchmark, get_libero_path

class DataLoader():
    def __init__(self, dataset_filenames, device="cpu", seq_len=1,
                 hdf5_cache_mode="low_dim"):
        # dataset_filenames maps task_id -> demo file, one entry per task to
        # train on. Single-task is the one-entry case, not a separate path.
        #
        # seq_len applies to obs AND actions (robomimic is built for BC-RNN).
        #   The eventual ACT plan needs a 16-length action window but only 1
        #   image; robomimic can't express that asymmetry, so revisit then.
        # hdf5_cache_mode="all" caches processed windows in host RAM -- measured
        #   ~2GB at seq_len=1 but ~30GB at seq_len=16. Only pair it with a small
        #   seq_len. "low_dim" caches proprio only and reads images from disk.
        #   That per-batch cost is now paid across N tasks, so "all" is even
        #   less affordable than it was.
        self.device = device

        obs_modality = {
            "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
            "low_dim": ["joint_states", "gripper_states"],
        }

        self.task_ids = list(dataset_filenames)
        self.datasets = []
        for task_id in self.task_ids:
            # Relative to the datasets root, including the problem folder:
            # "libero_spatial/<task>_demo.hdf5".
            dataset_file = os.path.join(
                get_libero_path("datasets"),
                dataset_filenames[task_id]
            )

            dataset, shape_meta = D.get_dataset(
                dataset_path=dataset_file,
                obs_modality=obs_modality,
                initialize_obs_utils=True,
                seq_len=seq_len,
                frame_stack=1,
                hdf5_cache_mode=hdf5_cache_mode,
            )

            self.datasets.append(dataset)
            self.shape_meta = shape_meta

        # Flat index across the concatenated datasets, so a sample's chance of
        # being drawn is uniform over transitions. Sampling the task first and
        # then within it would over-weight whichever task has the shortest
        # demos.
        lengths = np.array([len(d) for d in self.datasets])
        self.starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self.total = int(lengths.sum())

    def __len__(self):
        return self.total

    def get_batch(self, batch_size=64, device=None):
        # device=None -> use the DataLoader's configured self.device.
        device = self.device if device is None else device
        # Sampled with replacement -- replay-buffer idiom, no epoch boundaries.
        # Index [0] of each window: obs paired with the action at that same
        # step. At seq_len > 1 the extra frames are cached but discarded here.
        flat = np.random.randint(0, self.total, size=batch_size)
        which = np.searchsorted(self.starts, flat, side="right") - 1
        local = flat - self.starts[which]

        agentview, wrist, joint_state, actions, task_id = [], [], [], [], []
        for w, i in zip(which, local):
            d = self.datasets[w][i]
            agentview.append(d["obs"]["agentview_rgb"][0])
            wrist.append(d["obs"]["eye_in_hand_rgb"][0])
            joint_state.append(np.concatenate([
                d["obs"]["joint_states"][0],
                d["obs"]["gripper_states"][0],
            ]))
            actions.append(d["actions"][0])
            # The raw suite task_id, matching what the model's embedding and the
            # live env are indexed by.
            task_id.append(self.task_ids[w])

        def stack(arr):
            return torch.as_tensor(np.stack(arr)).float().to(device)

        return {
            "agentview": stack(agentview),   # (B, 3, 128, 128)
            "wrist":     stack(wrist),        # (B, 3, 128, 128)
            "joint_state":   stack(joint_state),      # (B, 9)
            "actions":   stack(actions),      # (B, 7)
            # Long, not float: it indexes an embedding.
            "task_id":   torch.as_tensor(np.array(task_id)).long().to(device),
        }
