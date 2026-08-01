from agent import Agent

agent = Agent()

# eval_every runs the same rollout loop eval.py uses, so training-time and
# standalone success rates cannot drift. Set to None to skip (rollouts are
# serial and cost real time).
agent.train(epochs=100000, batch_size=32, eval_every=10000, n_eval=20)
