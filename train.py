from agent import Agent

agent = Agent()

# eval_every runs the same rollout loop eval.py uses, so training-time and
# standalone success rates cannot drift. Set to None to skip (rollouts are
# serial and cost real time).
#   ~120 train steps/sec  -> 100k steps is ~14 min
#   eval every 10k steps  -> 10 points on the success-rate curve
#   n_eval=20             -> the noise floor; fewer rollouts is not a metric
agent.train(steps=100000, batch_size=32, eval_every=10000, n_eval=20)
