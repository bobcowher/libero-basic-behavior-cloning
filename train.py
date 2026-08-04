from agent import Agent


# The __main__ guard is REQUIRED, not stylistic: parallel eval uses the spawn
# start method, and spawned workers re-import this module. Without the guard
# each worker would start its own training run.
def main():
    # All 10 libero_spatial tasks. They share a scene and differ only in which
    # black bowl the instruction names, so this is only well-posed because the
    # model now takes a task embedding -- drop that and the ten tasks become the
    # same observation with contradictory labels.
    agent = Agent(task_ids=list(range(10)))

    # eval_every runs the same rollout loop eval.py uses, so training-time and
    # standalone success rates cannot drift.
    #   ~120 train steps/sec -> 100k steps is ~14 min
    #   eval now costs 10x a single-task eval, one env batch per task. At
    #     n_eval=5, eval_env_num=5 that is 10 batches, ~4 min measured against
    #     the 29s/batch from the 50-scene rescores -- so eval_every=20000
    #     (5 curve points) keeps the run near its old ~1h wall clock.
    #   n_eval=5 x 10 tasks = 50 rollouts per curve point, 5x what the
    #     single-task runs averaged over. Per-task curves are jumpy at 5 scenes;
    #     they are diagnostic, the mean is the metric.
    #   save_every=10000 overwrites the checkpoint unconditionally, so the file
    #     on disk is always the latest weights. Best-loss selection is gone --
    #     run 427 showed it keeps a model that scores worse than ones it threw
    #     away, because loss keeps falling long after success rate flattens.
    agent.train(steps=100000, batch_size=32, eval_every=20000,
                n_eval=5, eval_env_num=5, save_every=10000)


if __name__ == "__main__":
    main()
