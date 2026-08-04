"""Smoke test the parametrized Model: forward pass across sweep configs."""
import torch
from model import Model

img_shape = (3, 128, 128)
joint_dim = 9
n_actions = 7
n_tasks = 10
b = 4

configs = [
    dict(hidden_dim=256, compression_dim=256, n_hidden_layers=1),  # baseline
    dict(hidden_dim=64, compression_dim=64, n_hidden_layers=1),
    dict(hidden_dim=1024, compression_dim=1024, n_hidden_layers=1),
    dict(hidden_dim=256, compression_dim=256, n_hidden_layers=2),
    dict(hidden_dim=256, compression_dim=64, n_hidden_layers=1),
    dict(hidden_dim=256, compression_dim=1024, n_hidden_layers=1),
]

img = torch.zeros(b, *img_shape)
joints = torch.zeros(b, joint_dim)
tasks = torch.arange(b) % n_tasks

for c in configs:
    m = Model(image_input_shape=img_shape, joint_input_dim=joint_dim,
              num_actions=n_actions, n_tasks=n_tasks,
              checkpoint_dir="/tmp/smoke_ckpt", **c)
    out = m(img, joints, tasks)
    n_params = sum(p.numel() for p in m.parameters())
    assert out.shape == (b, n_actions), out.shape
    assert torch.isfinite(out).all()
    print(f"OK  {c}  ->  out {tuple(out.shape)}  params {n_params:,}")

# The task id must actually reach the output. Same image, same proprio, two
# task ids -- if these agree the embedding is wired in but doing nothing, which
# is exactly the failure that would look like "multi-task trains fine" while the
# policy stays task-blind.
m = Model(image_input_shape=img_shape, joint_input_dim=joint_dim,
          num_actions=n_actions, n_tasks=n_tasks, hidden_dim=256,
          checkpoint_dir="/tmp/smoke_ckpt")
a = m(img, joints, torch.zeros(b, dtype=torch.long))
z = m(img, joints, torch.full((b,), 7, dtype=torch.long))
assert not torch.allclose(a, z), "task embedding does not change the output"
print(f"OK  task id changes output (mean |delta| {(a - z).abs().mean():.4f})")

print("all configs passed")
