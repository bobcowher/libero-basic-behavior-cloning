"""Batch-size ablation: does a bigger batch buy anything on libero_spatial task 0?

Question: run 427 drove train loss to 0.0058 while rollout success peaked at 20%
and then went flat. Optimization is not the bottleneck. Raising the batch size
lowers gradient noise -- which may help throughput, but also removes one of the
few regularizers in a 5068-transition dataset. This measures it instead of
arguing about it.

Design:
  - **Matched samples seen**, not matched steps: 50000 x 32 == 6250 x 256 ==
    1.6M samples. Comparing at equal steps would just hand the big-batch arm 8x
    the compute.
  - **sqrt LR scaling** (1e-4 -> 2.83e-4 for an 8x batch). Adam already
    normalizes gradient magnitude, so linear scaling over-corrects on a dataset
    this small.
  - **Identical seed** for both arms, so weight init and env init states match.
  - In-loop eval points land on identical sample counts (every 320k samples), so
    the two curves are directly comparable.
  - The headline number is a **50-scene eval of the final model**, taken
    in-process. Scoring the saved checkpoint instead would confound the
    comparison with best-loss checkpoint selection, which run 427 showed picks
    the wrong weights.
  - **Separate ckpt names per arm.** Sharing one file is how this project has
    lost checkpoints twice.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from agent import Agent

SAMPLES = 1_600_000          # 50000 * 32
EVAL_EVERY_SAMPLES = 320_000 # 5 curve points per arm, at matched sample counts
SEED = 0

ARMS = [
    # (tag, batch_size, lr)
    ("b32",  32,  1e-4),
    ("b256", 256, 2.83e-4),   # 1e-4 * sqrt(256/32)
]


def run_arm(tag, batch_size, lr):
    steps = SAMPLES // batch_size
    eval_every = EVAL_EVERY_SAMPLES // batch_size

    print("=" * 70, flush=True)
    print(f"ARM {tag}: batch={batch_size} lr={lr:g} steps={steps} "
          f"eval_every={eval_every} (samples seen={steps * batch_size:,})",
          flush=True)
    print("=" * 70, flush=True)

    torch.manual_seed(SEED)
    agent = Agent(lr=lr, ckpt=f"checkpoints/ablation_{tag}")

    t0 = time.time()
    agent.train(steps=steps, batch_size=batch_size, eval_every=eval_every,
                n_eval=10, eval_env_num=10, run_tag=f"ablation-{tag}")
    train_secs = time.time() - t0

    # Full test set, full horizon, on the FINAL weights (not the saved ones).
    print(f"[{tag}] scoring final model on all 50 scenes...", flush=True)
    result = agent.evaluate(n_eval=50, max_steps=600, env_num=10)
    print(f"[{tag}] FINAL {result}   (train {train_secs / 60:.1f} min)",
          flush=True)
    return tag, batch_size, lr, train_secs, result


def main():
    results = [run_arm(*arm) for arm in ARMS]

    print()
    print("=" * 70)
    print("BATCH ABLATION RESULTS  (1.6M samples seen per arm, 50-scene eval)")
    print("=" * 70)
    print(f"{'arm':>6}  {'batch':>6}  {'lr':>9}  {'train min':>9}  {'success':>9}  solved")
    for tag, batch_size, lr, secs, r in results:
        solved = sum(r.successes)
        print(f"{tag:>6}  {batch_size:>6}  {lr:>9.2e}  {secs / 60:>9.1f}  "
              f"{r.success_rate:>8.1%}  {solved}/50")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
