# Full-track decoder evaluation

From the repository root, use a fresh output directory for each run.

```powershell
python scripts/run_route_evaluation.py --data-root "PATH_TO_DATASET" --output experiments/person3/route_phase2_smoke --smoke
python scripts/run_route_evaluation.py --data-root "PATH_TO_DATASET" --output experiments/person3/route_phase2
```

The smoke run uses the first four sorted training trip IDs. The full run uses the entire training roster. Two row-balanced GroupKFold partitions hold out complete trips. Each fold fits the original fixed Extra Trees and HistGradientBoosting configurations on the 19-column training matrix and persists and reloads the fitted models. The manifest identifies which trip IDs each artifact can evaluate out of fold. Loading a model checks its feature order and binary class order; those checks alone cannot establish its training provenance.

Inference reads only the held-out trip's track, nodes and arcs, builds the candidate table for the entire trip and applies the existing nine baseline and ten additional features. Independent predictions use the first maximum-probability baseline candidate, consistent with the classifier experiments. Viterbi uses the Phase 1 defaults and its own deterministic arc-ID tie policy. Offline motion features can use the following GPS fix. Reference alignment rows are joined by trip and observation ID after inference for both folds is finished.

## Reading the outputs

- `summary.csv` pools held-out observations across both folds. Conditional Top-1 and trip-qualified Macro F1 share a single mask for both methods: a valid aligned reference arc present in the generated candidate set. All-valid Top-1 counts a candidate miss as incorrect. Candidate recall divides the matched count by the valid-reference count. Prediction coverage is the proportion of **all raw fixes** receiving an arc.
- Transition validity measures physically feasible selected transitions over all adjacent populated fixes at most five seconds apart, including transitions across decoder resets. Both methods use the same directed metric, along-arc offsets and speed/slack limit. No reference labels enter this calculation.
- `latencies.csv` gives seconds per held-out trip. Preparation includes raw file I/O, graph, candidate generation, and features, shared by both models. Emission timing includes prediction, construction of observation states, and independent choice. Decode timing is the additional Viterbi pass. Evaluation, persistence and offline classifier fitting are outside these inference timings.
- The compressed observation CSVs retain predictions, references, coverage mask, transition checks and decoder status for auditing paired comparisons. Null predictions indicate empty states. The output describes road arcs at observed fixes. It does not measure the accuracy of unsampled connector arcs.
