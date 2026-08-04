"""End-to-end multi-task check: dataloader -> model -> live env -> eval.

Two tasks, not ten, so it runs in about a minute. Builds real envs and reads
real HDF5s, because the failures worth catching here are exactly the ones a
mocked test cannot see: an id that means one thing in the loader and another in
the env.

    PYTHONPATH=. python scripts/smoke_multitask.py
"""
import numpy as np
import torch

from agent import Agent

TASKS = [0, 7]


# The __main__ guard is REQUIRED: env_num > 1 evaluates through
# SubprocVectorEnv with the spawn start method, and spawned workers re-import
# this module. Without it they re-run the script instead of the worker body.
def main():
    agent = Agent(task_ids=TASKS)
    print(f"tasks {agent.task_ids}  n_tasks(embedding) {agent.n_tasks}")
    for t in agent.task_ids:
        print(f"  [{t}] {agent.tasks[t].language}")

    # --- dataloader --------------------------------------------------------
    batch = agent.dl.get_batch(batch_size=64)
    tid = batch["task_id"]
    assert tid.dtype == torch.long, tid.dtype
    assert set(tid.unique().tolist()) <= set(TASKS), tid.unique()
    # Both tasks must actually appear. Flat sampling over ~10k transitions makes
    # a 64-sample batch missing one of two tasks a ~1e-19 event, so this failing
    # means the index mapping collapsed, not bad luck.
    assert len(tid.unique()) == 2, f"only saw tasks {tid.unique().tolist()}"
    print("batch task_id counts: "
          f"{ {t: int((tid == t).sum()) for t in TASKS} }")

    # Sampling is uniform over transitions, so the split tracks dataset sizes.
    sizes = [len(d) for d in agent.dl.datasets]
    print(f"dataset sizes {sizes}  total {len(agent.dl)}")

    # --- model -------------------------------------------------------------
    out = agent.model(batch["wrist"], batch["joint_state"], tid)
    assert out.shape == (64, agent.action_dim), out.shape
    assert torch.isfinite(out).all()
    print(f"forward OK -> {tuple(out.shape)}")

    # --- live env, both tasks ----------------------------------------------
    # The real point of this script: _act must produce a different action for
    # the same live obs under a different task id, and every task's env must
    # build and accept the policy's output.
    for t in TASKS:
        other = TASKS[1] - t + TASKS[0]
        env = agent._build_env(1, seed=0, task_id=t)
        try:
            env.reset()
            obs = env.set_init_state(agent._init_states(t)[np.array([0])])
            for _ in range(agent.settle_steps):
                obs = env.step(np.zeros((1, agent.action_dim)))[0]
            a_self = agent._act(obs, t)
            a_other = agent._act(obs, other)
            assert a_self.shape == (1, agent.action_dim), a_self.shape
            assert np.isfinite(a_self).all()
            delta = np.abs(a_self - a_other).mean()
            assert delta > 0, "same action regardless of task id"
            print(f"task {t}: env OK, action {np.round(a_self[0], 3)}, "
                  f"mean |delta vs task {other}| {delta:.4f}")
        finally:
            env.close()

    # --- eval path ---------------------------------------------------------
    results = agent.evaluate_tasks(n_eval=2, max_steps=30, env_num=2)
    assert set(results) == set(TASKS), results.keys()
    for t, r in results.items():
        print(f"eval task {t}: {r}")

    print("multi-task smoke passed")


if __name__ == "__main__":
    main()
