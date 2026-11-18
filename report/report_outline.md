# PathSnap technical report outline

**Purpose:** Shared writing structure for the team. Replace every bracketed prompt with the team's own method, evidence, and measured results. Do not submit placeholders or unrun results.

## Cover page

- Project: PathSnap — Distributed Map-Matching and Route Reconstruction
- Course / section: [fill in]
- Team members: [names and roles]
- Repository: [public GitHub URL]
- Submission date: [date]

## 1. Problem and system overview

- What GPS map-matching problem does PathSnap solve?
- Summarize the end-to-end pipeline: raw data → road candidates → features → classifier → HMM/Viterbi → API/UI.
- Include a compact architecture diagram, especially the two-laptop deployment.
- **Owner:** Person 4 drafts system architecture; all members review.

## 2. Dataset understanding and exploratory analysis

### 2.1 Data partitions and file formats

- State trip counts and permitted use of train, student-test, and evaluator-test partitions.
- Explain `.track`, `.nodes`, `.arcs`, and `.route` using the observed schema.
- State that evaluator-testing data and labels were excluded from development.
- **Owner:** Person 1.

### 2.2 Data quality and geographic extent

- Include trip/file inventory and row-count distributions.
- Report missing/malformed rows, coordinate ranges, timestamp order, sampling intervals, and large-gap cases.
- Report road-network extents, node/arc counts, isolated-node counts, weakly connected components, and largest-component fractions.
- Include the actual plots from `results/plots/` and describe the partition used.
- Explain the distance calculation/projection used and why it fits the dataset's geographic scale.
- **Owner:** Person 1; Person 2 contributes geographic maps.

### 2.3 Preprocessing, route labels, and assumptions

- Explain how raw files are loaded and validated.
- Explain how reference route IDs are aligned to GPS observations; do not claim row-for-row alignment.
- Document alignment method, parameters, candidate coverage/miss rate, ambiguous cases, and manual inspection procedure.
- State limitations: reference routes are not assumed error-free independent ground truth.
- **Owner:** Person 1 drafts alignment; Person 2 and Person 3 review candidate/feature assumptions.

## 3. Candidate generation and feature engineering

### 3.1 Road-segment candidates

- Define how nearby arcs are found, candidate radius/top-k settings, coordinate projection/distance metric, and handling of empty candidate sets.
- Report candidate miss rate using the team's label-alignment method.
- **Owner:** Person 2.

### 3.2 Features

- List every feature, units, computation, and reason for inclusion.
- Explain missing-value and scaling choices and how preprocessing is fitted within each training fold.
- **Owner:** Person 3; Person 1 documents raw-data fields used.

## 4. Model experimentation and comparison

### 4.1 Experimental protocol and metrics

- Describe trip-grouped cross-validation on the 80 training trips and use of the 10 student-test trips for development-stage comparison.
- Define observation-level Top-1 Accuracy, Macro F1, candidate-level Log Loss, candidate miss rate, route continuity, training time, and inference time.
- Explain class keys: `trip_id::candidate_arc_id`, so local arc IDs from different trips are not conflated.
- State random seeds and actual preprocessing/model configuration.
- **Owner:** Person 3.

### 4.2 Overall model comparison table

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 Accuracy | Student-test Macro F1 | Route continuity | Log Loss | Inference time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Linear | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |
| Gaussian Naive Bayes | Probabilistic | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |
| Decision Tree | Tree | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |
| Random Forest | Ensemble | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |
| Extra Trees | Ensemble | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |
| HistGradientBoosting | Boosting | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |
| KNN | Instance-based | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |
| MLP | Neural network | [actual] | [actual] | [actual] | [actual] | [actual/N/A + reason] | [actual] | [actual] |

Only include models actually trained. Explain N/A values.

### 4.3 Individual model subsections

Create one subsection per model actually trained. For each, answer:

1. Why was it suitable for candidate-road selection?
2. What strengths and limitations did it show for this task?
3. What were the important initial hyperparameters and configuration?
4. What loss/objective did it optimize?
5. What were its actual cross-validation and student-test results?
6. What evidence indicated overfitting, underfitting, or unstable performance?

Include actual plots: CV Macro F1 comparison, student-test Top-1 and Macro F1 comparison, and Log Loss comparison where applicable. Include training curves only for models that produced real epoch-wise histories. Confusion matrices are optional diagnostics.

### 4.4 Model selection

- Identify the selected model and the primary metric used.
- Support the choice with measured generalization, route continuity, probability quality, inference cost, and compatibility with decoding.
- Discuss limitations and cases where the selected model did not perform well.
- **Owner:** Person 3 leads; each model owner supplies their own actual results.

## 5. Bayesian hyperparameter optimization

- Selected model: [actual]
- Search parameters and ranges/distributions: [actual]
- Objective and grouped CV strategy: [actual]
- Number of trials, search record, and compute cost: [actual]
- Best values: [actual]
- Compare tuned vs. untuned using the same CV/student-test partitions and metrics.
- **Owner:** Person 3.

## 6. HMM and Viterbi route reconstruction

- Define hidden states and how candidate-to-arc IDs are preserved.
- Explain emission and transition probabilities, route connectivity, missing candidates, large GPS gaps, and disconnected transitions.
- Describe the Viterbi recurrence or implementation.
- Compare classifier-only with classifier-plus-decoder using actual pointwise metrics and sequence/route-continuity measures.
- Discuss cases where decoding helped and cases where it did not.
- **Owner:** Person 3.

## 7. Geographic visualizations

- Include at least one representative Folium map generated by the submitted code.
- Show raw GPS observations, road network, reference route where available, and predicted route in distinguishable layers/colors.
- Include labels/legend, suitable extent/zoom, and layer controls where appropriate.
- Include a screenshot/export in this PDF and commit the interactive HTML or generating code.
- Identify the trip and partition for every map; do not use evaluator labels in development maps.
- **Owner:** Person 2; Person 4 can capture UI route displays.

## 8. API, UI, and deployment

- Explain API endpoints, model loading, request/response format, and two-laptop deployment steps.
- Include screenshots of the working UI and successful API response/logs on the receiving laptop.
- Confirm evaluator ground-truth labels remain server-side and are not sent into the prediction model or UI.
- **Owner:** Person 4.

## 9. Limitations, reproducibility, and conclusion

- Summarize data limitations, alignment uncertainty, candidate misses, model/decoder limitations, and known failure cases.
- Give exact repository instructions to install, train, evaluate, start the API, start the UI, and regenerate plots/maps.
- Summarize conclusions based only on actual evidence.
- **Owners:** All members; Person 4 consolidates.

## 10. AI assistance and academic integrity disclosure

- Disclose AI assistance according to course/event policy.
- Ensure every written claim, plot, metric, and citation can be verified from the repository and explained by the team.
- Never report fabricated experiments or results.
- **Owners:** All members.

## Appendix: reproducibility evidence

- Repository commit/revision used for the report.
- Environment and dependency versions.
- Random seeds and configuration files.
- Experiment logs and paths to generated plots/tables.
- Data partition used for each result.
