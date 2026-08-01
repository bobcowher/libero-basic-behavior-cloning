"""Score a checkpoint by rollout success rate.

    python eval.py --ckpt checkpoints/bc_network --n-eval 50
    python eval.py --zero-action --n-eval 3      # smoke test, no checkpoint

Success rate is the only real BC metric -- validation loss is nearly
uncorrelated with it.
"""
import argparse

from agent import Agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/bc_network")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=50,
                   help="rollouts; fewer than 20 is noise")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--env-num", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-action", action="store_true",
                   help="ignore the model, send zeros; must score 0%%")
    args = p.parse_args()

    agent = Agent(task_id=args.task, ckpt=args.ckpt)
    print(f"task {args.task}: {agent.task.language}")

    if not args.zero_action:
        agent.load_checkpoint()

    result = agent.evaluate(
        n_eval=args.n_eval,
        max_steps=args.max_steps,
        env_num=args.env_num,
        seed=args.seed,
        zero_action=args.zero_action,
    )
    print(result)


if __name__ == "__main__":
    main()
