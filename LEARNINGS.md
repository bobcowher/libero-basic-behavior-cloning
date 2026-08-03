# Learnings Log

A running record of experiments and what they told us. Newest entries on top.

---

## 2026-08-02 — Architecture sweep: layer size is not the bottleneck

**Setup.** Parametric sweep over network size on the wrist-camera + joint-state BC
policy. Model was parametrized so each config is a 3-number diff: `hidden_dim`,
`compression_dim` (per-modality embedding width, previously tied to `hidden_dim`),
and `n_hidden_layers` (fully-connected layers between fusion and output). 15 training
runs total (Beekeeper runs 434–449), each on its own `exp-*` branch off `exp-base`.
Nothing was merged to main during the sweep.

**Metric.** Primary: `eval/success_rate` (higher better). Eval is a single 10-scene
pass at fixed seed 0, giving success resolution 0.10 and a ~±0.20 noise band. Because
training init and batch order are unseeded, re-running a branch is a valid new training
seed on the same fixed eval scenes — that's how we got multi-seed replicates.

**Results.**

| Config              | Seeds | Mean success | Range      | Notes                        |
|---------------------|:-----:|:------------:|:-----------|:-----------------------------|
| hidden 1024         |   3   | 0.43         | .30–.50    | top mean, peaks to .80       |
| baseline 256/256/1  |   3   | 0.40         | .30–.60    | the starter                  |
| extra layer 256/L2  |   3   | 0.40         | .40–.40    | zero spread                  |
| comp 1024 (n=1)     |   1   | 0.50         | peak 0.70  | wildcard, one seed only      |
| hidden 512          |   1   | 0.30         | peak 0.40  | single seed                  |
| comp 64/128/512     |   1   | 0.30–0.40    | —          | within noise                 |
| hidden 128          |   1   | 0.10         | peak 0.30  | underfit                     |
| hidden 64           |   1   | 0.20         | peak 0.40  | underfit                     |

**What we learned.**

1. **Capacity is a plateau, not a slope.** Everything ≥256 is a statistical dead
   heat — means 0.40–0.43 with fully overlapping seed ranges, all smaller than the
   ±0.20 eval noise. The 12.7M-param 1024 net ties the 2.6M-param baseline.
2. **There's a floor at ~256, no ceiling above it.** Narrow nets (64, 128) clearly
   underfit and are the only unambiguous losers.
3. **The old "256 = 0.60" was seed luck.** Re-running the same code gave 0.40 / 0.50 /
   0.30. Multi-seed replication caught exactly the noise it was meant to.
4. **`train/l1_loss` is an anti-signal for ranking.** The 1024 nets had the lowest
   loss (~0.0035) and did not win on success rate. Rank by `success_rate` only.
5. **The bottleneck is data/eval, not the skeleton.** When width, depth, and
   compression all fail to move the number, the model is not the limiter.

**Decision.** Keep **256/256/1** as the working skeleton — tied with everything bigger,
cheapest and fastest to iterate. Real gains come from data, eval design, and multi-goal
work, not from resizing layers. This was architecture-scouting, not model production.

**Parked follow-ups.**
- comp-1024 flashed 0.50 final / 0.70 peak on a single seed — the one open wildcard,
  worth 2 more seeds only if we decide to chase a ceiling.
- Best-success checkpointing: `save_checkpoint()` currently overwrites with the *latest*
  weights every 10k steps, so a run's best in-run eval (e.g. comp-1024's 0.70 @ 70k) is
  discarded. Per-run persistent dirs keep each run's final model, not its intra-run best.
