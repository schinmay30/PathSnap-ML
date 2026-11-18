# Decoder policy evaluation

The original `decode(graph, observations)` defaults reproduce Phase 2. Additional keyword arguments are optional.

| Parameter | Default | Effect |
|---|---:|---|
| emission_weight | 1 | Multiplies each log classifier probability. |
| transition_weight | 1 | Multiplies both travel penalties and any soft-jump cost. |
| distance_deadband_m | 0 | Ignores this many metres of absolute network/GPS displacement mismatch before dividing by distance_scale_m. |
| displacement_reset_m | None | Starts a new segment when observed displacement exceeds this threshold. Time-gap resets take precedence. |
| unreachable_penalty | None | Default disallows travel beyond the speed/slack budget or without a directed route. A finite nonnegative value allows either as a penalized link and marks it soft_jump if selected in the final path. A soft jump is a discontinuity, not a drivable connector. |

The existing gap_seconds, max_speed_mps, gps_slack_m and distance_scale_m remain configurable. Empty fixes always segment the path. Both physical validity and latency use the Phase 2 measurement definitions; a selected soft jump remains invalid at the common physical threshold.

## Experiment design

The budget is six declared settings per emission model per outer fold. In declaration order these are Phase 2, quarter transition weight, quarter weight with a 10 m deadband, a 65 m displacement reset, a soft-jump cost of 2, and that same soft-jump cost with triple emission weight. Each comparison changes one setting relative to its parent. The classifier hyperparameters, 19 features and candidate generator are fixed.

Within each of the two original outer folds, eight of the 40 development trips are sampled uniformly without replacement (NumPy seed 42 plus fold number). Each emitter fits the other 32 trips. The policy with highest pooled eligible-observation Top-1 on the eight validation trips wins, breaking ties by physical validity then declaration order. Each emitter is refit on all 40 development trips before scoring the winning policy on the other 40. This is one inner holdout per outer fold. The outer independent predictions, reference IDs and masks must match saved Phase 2 rows. The baseline is previously inspected development evidence.

```powershell
python scripts/run_decoder_ablation.py --data-root "PATH_TO_DATASET"
python scripts/diagnose_route_resets.py --data-root "PATH_TO_DATASET" --output experiments/person3/decoder_ablation/phase2_diagnosis.csv
```

Inner and outer stage timings include shared graph/feature preparation, emissions, and decoding as separate components. Preparation is performed once per trip and should only be counted once per model/policy latency estimate. Per-trip comparisons use paired rows in per_trip.csv. All-valid Top-1 uses every valid route-derived label, including candidate misses. Other metrics follow docs/route_evaluation.md.
