"""Score a checkpoint by rollout success rate.

    python eval.py --n-eval 50                   # newest checkpoint on disk
    python eval.py --ckpt checkpoints/bc_network --n-eval 50
    python eval.py --zero-action --n-eval 3      # smoke test, no checkpoint

Success rate is the only real BC metric -- validation loss is nearly
uncorrelated with it.
"""
import argparse
import os

from agent import Agent

CKPT_DIR = "checkpoints"


def latest_ckpt():
    """Newest file in checkpoints/ by mtime.

    Covers both sources without a second mechanism: training writes there, and
    download_models.sh drops run-tagged files there, so "newest" is whichever
    model you most recently trained or pulled.
    """
    if os.path.isdir(CKPT_DIR):
        files = [os.path.join(CKPT_DIR, f) for f in os.listdir(CKPT_DIR)]
        files = [f for f in files if os.path.isfile(f)]
        if files:
            return max(files, key=os.path.getmtime)
    return os.path.join(CKPT_DIR, "bc_network")


def parse_scenes(spec):
    """"0,10,20" or "0-9" or a mix -> [0, 10, 20] / [0..9].

    Shared with test.py so the two entry points name scenes identically. That
    is what makes `eval.py --scenes 0,10,20` the direct check on what
    `test.py --scenes 0,10,20` just showed you.
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="default: newest file in checkpoints/")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=50,
                   help="rollouts; fewer than 20 is noise")
    p.add_argument("--scenes", default=None,
                   help='init states to score, e.g. "0-9" or "0,10,20"; '
                        "overrides --n-eval")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--env-num", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-action", action="store_true",
                   help="ignore the model, send zeros; must score 0%%")
    args = p.parse_args()

    ckpt = args.ckpt or latest_ckpt()

    agent = Agent(task_id=args.task, ckpt=ckpt)
    print(f"task {args.task}: {agent.task.language}")
    # Always name the weights being scored. A silently-chosen checkpoint is how
    # you end up attributing one model's number to another.
    print(f"checkpoint: {ckpt}")

    if not args.zero_action:
        agent.load_checkpoint()

    result = agent.evaluate(
        n_eval=args.n_eval,
        scenes=parse_scenes(args.scenes) if args.scenes else None,
        max_steps=args.max_steps,
        env_num=args.env_num,
        seed=args.seed,
        zero_action=args.zero_action,
    )
    print(result)
    # Which scenes, not just how many. A single scene replayed looks like a
    # working policy; this names the ones that actually work, so you can watch
    # them with `test.py --scenes N`.
    print(f"solved scenes: {result.solved}")


if __name__ == "__main__":
    main()
