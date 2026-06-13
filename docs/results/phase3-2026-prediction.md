# Phase 3 Headline Result — 2026 World Cup Title Probabilities

**Date:** 2026-06-13
**Run:** `python scripts/predict_2026.py data/raw/results.csv`
**Method:** M1 dynamic hierarchical Poisson fit via **NUTS** (2 chains × 1000 draws, 500 tune; 82s) on the last 12 years of international results → ~2000 posterior strength draws at the latest period → **20,000 posterior-propagated, correlated** simulations of the real 48-team 2026 bracket (groups derived from the actual fixture list; the 4 already-played group matches held fixed; hosts USA/Canada/Mexico given home advantage).

## Top 20 (of 48)

```
       team  p_win  p_final   p_sf   p_qf  p_r16  p_qualify
  Argentina 0.2034   0.3090 0.4791 0.6153 0.7789     0.9736
      Spain 0.1197   0.2072 0.3129 0.4810 0.7351     0.9812
     Brazil 0.1090   0.1779 0.2866 0.4462 0.6663     0.9805
    England 0.0743   0.1386 0.2361 0.4380 0.6662     0.9656
     France 0.0585   0.1187 0.2382 0.3846 0.5934     0.9185
   Portugal 0.0561   0.1106 0.2014 0.3674 0.5905     0.9238
   Colombia 0.0544   0.1035 0.1924 0.3508 0.5800     0.9192
    Germany 0.0360   0.0810 0.1599 0.3428 0.6266     0.9704
    Belgium 0.0354   0.0761 0.1583 0.2961 0.6410     0.9297
Netherlands 0.0339   0.0835 0.1799 0.3273 0.5634     0.8810
     Mexico 0.0267   0.0660 0.1480 0.3014 0.6163     0.9879
    Morocco 0.0236   0.0546 0.1148 0.2286 0.4686     0.8950
    Uruguay 0.0205   0.0474 0.1060 0.2198 0.4294     0.8950
      Japan 0.0194   0.0516 0.1252 0.2521 0.4712     0.8313
    Ecuador 0.0180   0.0450 0.1041 0.2415 0.4898     0.9052
Switzerland 0.0163   0.0411 0.0977 0.2301 0.5723     0.9452
     Norway 0.0131   0.0360 0.0940 0.2073 0.4026     0.7939
    Croatia 0.0121   0.0350 0.0817 0.2110 0.4254     0.8892
    Senegal 0.0099   0.0283 0.0746 0.1712 0.3572     0.7491
    Austria 0.0095   0.0232 0.0656 0.1855 0.3789     0.7518
```

`sum(p_win) = 1.0000` across all 48 teams (exactly one champion per simulation).

## Sanity checks (all pass)

- **Probability mass conserved:** title probabilities sum to 1.0.
- **Monotone reach-rounds:** for every team `p_qualify ≥ p_r16 ≥ p_qf ≥ p_sf ≥ p_final ≥ p_win`.
- **No runaway favourite:** Argentina tops at 20.3% — at the upper edge of the spec's expected
  ~20–25% ceiling for a single-elimination favourite. Realistic, not degenerate.
- **Plausible contender ordering:** Argentina (reigning champion, top strength) → Spain → Brazil
  → England → France → Portugal — matches the consensus elite tier. Colombia (5.4%) ranks a touch
  high, reflecting strong recent form in the data; flagged but plausible.

## Honest caveats (carried from the spec)

- This is **not** a claim that Argentina will win — it is a ~1-in-5 chance, i.e. ~4-in-5 *against*.
  Single-elimination variance dominates; the spread across ~10 credible contenders is the real
  message.
- Two documented modelling approximations affect specific early matchups (not materially the
  champion distribution): group tiebreakers use points→GD→GF→random (head-to-head simplified),
  and the 8 best third-placed teams fill R32 slots by a fixed rule rather than FIFA's exact
  combination table.
- The in-sim engine is **M1** (it alone exposes the posterior needed for correlated draws). M1
  trailed the meta-learner on point RPS in the Phase-2 backtest (0.2064 vs 0.1999) but provides
  the best calibration and the uncertainty propagation the simulator requires.

## What this enables (Phase 4)

Re-running this after each 2026 matchday (re-fit conditioned on new real results) produces a
**title-odds timeline**; the web dashboard visualises this table + the bracket + calibration.
