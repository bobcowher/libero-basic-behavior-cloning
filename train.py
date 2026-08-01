from agent import Agent


# The __main__ guard is REQUIRED, not stylistic: parallel eval uses the spawn
# start method, and spawned workers re-import this module. Without the guard
# each worker would start its own training run.
def main():
    agent = Agent()

    # eval_every runs the same rollout loop eval.py uses, so training-time and
    # standalone success rates cannot drift.
    #   ~120 train steps/sec -> 100k steps is ~14 min
    #   eval every 10k steps -> 10 curve points (9 in-loop + 1 final)
    #   n_eval=10, env_num=10 -> exactly one parallel batch, so an eval costs
    #     about as long as a single rollout. Scores scenes 0-9 of the 50
    #     available; `eval.py --n-eval 50` is the full-test-set number.
    agent.train(steps=100000, batch_size=32, eval_every=10000,
                n_eval=10, eval_env_num=10)


if __name__ == "__main__":
    main()
