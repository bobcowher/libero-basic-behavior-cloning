"""Watch the policy run. Writes one MP4 per rollout into videos/.

    python test.py                       # newest checkpoint, 5 rollouts
    python test.py --n-eval 20           # find a failure to stare at
    python test.py --ckpt checkpoints/run428_ablation_b256

Videos are named by outcome (rollout03_fail.mp4), so the failures are the ones
you open.

Not an on-screen viewer, deliberately. Per CONTEXT.md the viewer needs
MUJOCO_GL=glfw with `use_camera_obs=False`, which means no agentview images --
the policy has nothing to act on. Frames here are the raw agentview obs, i.e.
exactly what the model sees, which is the more useful thing to watch anyway.

For a success rate, use eval.py -- this shares its rollout loop but defaults to
a handful of episodes, not the full 50-scene test set.
"""
import argparse

from agent import Agent
from eval import latest_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="default: newest file in checkpoints/")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=5, help="rollouts to record")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out", default="videos")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-action", action="store_true",
                   help="ignore the model, send zeros")
    args = p.parse_args()

    ckpt = args.ckpt or latest_ckpt()

    agent = Agent(task_id=args.task, ckpt=ckpt)
    print(f"task {args.task}: {agent.task.language}")
    print(f"checkpoint: {ckpt}")

    if not args.zero_action:
        agent.load_checkpoint()

    # env_num=1: serial rollouts, so the videos come out in scene order and a
    # run this small gains nothing from workers.
    result = agent.evaluate(
        n_eval=args.n_eval,
        max_steps=args.max_steps,
        env_num=1,
        seed=args.seed,
        zero_action=args.zero_action,
        record_dir=args.out,
    )
    print(result)


if __name__ == "__main__":
    main()
