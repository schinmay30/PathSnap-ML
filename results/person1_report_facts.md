# Person 1 model facts (generated; do not edit by hand)

Source: `experiments/person1/results/`. Partitions: CV on the 80 `train` trips (GroupKFold by trip, 5 folds); final comparison on the 10 `student_test` trips. Re-generate with `python scripts/summarize_person1.py`.

## Logistic Regression

- Configuration (classifier step; features are standardised first): `{'C': 1.0, 'class_weight': 'balanced', 'l1_ratio': 0.0, 'max_iter': 1000, 'penalty': 'deprecated', 'random_state': 42, 'solver': 'lbfgs'}`
- Random state 42; scikit-learn 1.9.1
- CV (5 folds): Top-1 0.7876 ± 0.0158, Macro F1 0.6390 ± 0.0430, log loss 0.2634 ± 0.0377, fit time 1.36 s per fold
- Student test: Top-1 0.8061, Macro F1 0.6106, log loss 0.2356 (19131 eligible observations; 2 without the reference arc among candidates)
- Train (refit on all 80 trips, evaluated on the same trips): Top-1 0.7862, Macro F1 0.6413, log loss 0.2625
- Fit time on all training candidates 1.55 s; inference on student test 11.3 ms

## Gaussian Naive Bayes

- Configuration (classifier step; features are standardised first): `{'priors': None, 'var_smoothing': 1e-09}`
- Random state 42; scikit-learn 1.9.1
- CV (5 folds): Top-1 0.7771 ± 0.0155, Macro F1 0.6218 ± 0.0446, log loss 0.5740 ± 0.1774, fit time 0.41 s per fold
- Student test: Top-1 0.7908, Macro F1 0.5960, log loss 0.5794 (19131 eligible observations; 2 without the reference arc among candidates)
- Train (refit on all 80 trips, evaluated on the same trips): Top-1 0.7762, Macro F1 0.6239, log loss 0.5643
- Fit time on all training candidates 0.47 s; inference on student test 23.0 ms

## Learned parameters

Logistic Regression coefficients on standardised features (largest magnitude first):

- `dist_rel_m`: -2.660
- `dist_rank`: -2.109
- `heading_cos`: +1.173
- `speed_ms`: -0.181
- `arc_length_m`: +0.137
- `n_candidates`: -0.132
- `dist_m`: -0.097
- `along_fraction`: +0.050
- `moving`: -0.012

Gaussian NB: positive-class mean minus negative-class mean per standardised feature:

- `dist_rel_m`: -1.299
- `dist_m`: -1.274
- `dist_rank`: -1.173
- `heading_cos`: +0.857
- `n_candidates`: -0.831
- `arc_length_m`: +0.682
- `speed_ms`: +0.160
- `moving`: +0.013
- `along_fraction`: +0.008

## Questions the report must answer for each model (your own analysis goes here)

A1. Why is this algorithm suitable for PathSnap?  
A2. Strengths and limitations for this task (tie them to the numbers above)?  
B5. Important hyperparameters and initial values (see configuration).  
B6. Loss / objective used.  
C7-8. Student-test Top-1, Macro F1, log loss (see above).  
C9. Overfitting or underfitting, and the evidence (compare train, CV and student-test).  
