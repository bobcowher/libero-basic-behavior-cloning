# Learnings Log

A running record of experiments and what they told us. Newest entries on top.

---

## 2026-08-03 — 50-scene rescore: the 10-scene eval was misleading, not just noisy

**Setup.** Re-scored the architecture sweep's final checkpoints on the full
50-scene test set instead of the 10 scenes training evals use. Done in a
separate clone (`libero-bc-eval`) with `hidden_dim`/`compression_dim`/
`n_hidden_layers` exposed as eval flags, since the `exp-*` branches differ from
`exp-base` only in those three numbers. No retraining: the 50 init states are
fixed and rollouts are deterministic at fixed `env_num`, so this is the same
weights measured at 5x resolution. ~145s per checkpoint.

Only runs 440-449 survived — Beekeeper's `run_history_max_runs` is 10, so
434-439 (the hdim-64/128/512 and comp-64 seeds) were already evicted. The
underfit configs are gone; the contested ones are all here.

**Results.** 50-scene score of each run's final checkpoint.

| Config              | Seeds | 10sc final | **50sc mean** | 50sc range |
|---------------------|:-----:|:----------:|:-------------:|:-----------|
| baseline 256/256/1  |   3   |    0.40    |   **0.42**    | .38–.46    |
| hdim 1024           |   2   |    0.40    |     0.40      | .34–.46    |
| comp 512            |   1   |    0.30    |     0.38      | —          |
| extra layer 256/L2  |   2   |    0.40    |     0.36      | .34–.38    |
| comp 128            |   1   |    0.40    |     0.34      | —          |
| comp 1024           |   1   |    0.50    |     0.30      | —          |

**What we learned.**

1. **The comp-1024 wildcard is dead.** Best 10-scene final in the sweep (0.50)
   and best peak (0.70); worst 50-scene score of all ten checkpoints (0.30).
   The parked follow-up is closed — do not spend seeds on it.
2. **The plateau result holds, and 256/256/1 was the right keep.** Everything
   lands in 0.34–0.42 and the cheapest config is at the top of the range.
3. **The 10-scene number was anti-correlated with truth, not merely noisy.**
   Rank correlation with the 50-scene score is near zero, and the config that
   looked best came last. "Noisy" implies an unbiased estimate with wide error
   bars; this was worse than that.
4. **Most of the apparent seed variance was eval resolution.** The baseline's
   three seeds spread .30–.50 at 10 scenes and .38–.46 at 50. The training seed
   matters less than we credited; the measurement was doing the moving.
5. **Within-run thrash is as large as every between-config difference.** On the
   fixed, deterministic 10-scene set, run 447 went
   30/30/30/**60**/10/**0**/10/20/20/30 across consecutive checkpoints, and run
   444 hit 0.80 at 70k then finished at 0.50. Consecutive checkpoints of one
   model swing more than the configs differ from each other.
6. **So scoring the final checkpoint is a lottery ticket, not a measurement.**
   Combined with `save_checkpoint()` overwriting with the latest weights, every
   number in the 08-02 table is one draw from a distribution that wide. This is
   a systematic downward bias on the whole sweep, not a rounding error.

**Decision.** Keep 256/256/1 and stop tuning the skeleton — confirmed twice
now, at 5x resolution the second time. Next work is multi-task, which is also
the first setting where extra capacity would have something to do.

**Method note.** Any future config comparison needs eval resolution fixed
first. Ranking configs on a 10-scene final checkpoint cannot resolve
differences smaller than the within-run swing, which is ~0.4.

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
| baseline 256/256/1  |   3   | 0.40         | .30–.50    | the starter                  |
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

**Correction (2026-08-03).** The baseline range above read `.30–.60`; the three
seeds' finals were 0.30 / 0.40 / 0.50, so it is `.30–.50`. The 0.60 was run
443's *peak* at 40k, not any seed's final — the same number item 3 calls seed
luck.

**Parked follow-ups.**
- ~~comp-1024 flashed 0.50 final / 0.70 peak on a single seed~~ — **closed
  2026-08-03**: it scores 0.30 on 50 scenes, worst of the sweep. See the
  rescore entry above.
- Best-success checkpointing: `save_checkpoint()` currently overwrites with the *latest*
  weights every 10k steps, so a run's best in-run eval (e.g. comp-1024's 0.70 @ 70k) is
  discarded. Per-run persistent dirs keep each run's final model, not its intra-run best.
  The rescore quantified the cost: consecutive checkpoints of one run swing up
  to 0.6, so which weights get saved matters more than which config trained them.
