# Methodology

## Scope and data

SENTINEL-BC studies selective decisions on the historical Wisconsin Diagnostic
Breast Cancer dataset. UCI reports 569 observations and 30 numerical nucleus
morphology features extracted from fine-needle aspirate images. We load its
equivalent scikit-learn bundle. This system accepts measurements, not images.
The source does not consistently specify physical units. We preserve source
values and avoid unsupported mm/pixel labels.

Canonical labels are **0 = BENIGN** and **1 = MALIGNANT**. Scikit-learn's original
encoding is the reverse. `data.load_data` remaps it immediately, and tests check
counts and known rows. The sole positive-class probability is `p_malignant`.

## Fixed data roles

A single seed (42) and sequential stratified splits reserve the following roles:

| Role | Rows | Permitted use |
|---|---:|---|
| Fitting and cross-validation | 273 | Model search, preprocessing, early stopping, OOD fitting |
| Probability calibration | 57 | Sigmoid calibration of the selected fitted estimator |
| Conformalization | 57 | MAPIE conformity scores only |
| Safety policy development | 68 | Threshold selection, attribution background evaluation, controlled stress tests |
| Locked final test | 114 | One evaluation after all decisions freeze |

Exact row indices and class counts live in `results/full/partitions.json`.
The loader rejects exact duplicate feature rows before partitioning. The public
bundle does not expose patient linkage, so related samples cannot be ruled out.
No identifier or target-derived feature enters the models. Distribution and
correlation analysis uses fitting rows. No SMOTE or PCA is applied: correlated
measurements remain visible and transformations stay simple.

## Classical and neural models

Logistic regression and an RBF SVM search two C values in full mode. Random
forest searches two depth choices; XGBoost searches depths 2 and 3. Three fixed,
stratified folds keep each imputer and standard scaler inside its training
pipeline. Per-family winning settings determine the final comparison models.
The primary estimator maximizes the mean of CV accuracy and malignant recall.
Lower CV negative log-likelihood breaks ties. Test performance never selects it.

The Keras MLP has 64 and 32 ReLU units, L2 regularization (0.001), dropout rates
0.25 and 0.20, and a sigmoid malignant output. Adam (0.001) minimizes binary
cross-entropy. Each network reserves 18% of its current training fold for early
stopping, fits its imputer/scaler on the remaining inner training rows, restores
the best validation-loss weights, and stops after 12 unimproved epochs or 100
epochs. The final networks likewise retain an internal validation split; they
are not claimed to fit all 273 rows. No BatchNorm layer is used.

Full mode trains independent seeds 42, 142 and 242, evaluates the single first
network and their mean-probability deep ensemble in the same CV protocol, and
trains final members from scratch. Quick mode uses fewer epochs, one network,
logistic regression and SVM. Its outputs occupy separate directories.

## Calibration and conformal prediction

`CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")` fits only on
the 57 calibration rows. Base weights remain frozen. All classes stay ordered
`[0, 1]`. Calibration is prespecified rather than chosen on the test set, and
may worsen finite-sample accuracy or calibration scores. Results explicitly
compare Brier score, NLL, ECE and reliability curves before and after calibration.
Temperature and isotonic calibration are omitted to avoid unnecessary fitting
on this small calibration sample.

MAPIE 1.2.0 `SplitConformalClassifier` wraps the calibrated estimator with
`prefit=True`, LAC conformity scores and 90% nominal confidence. It calls
`conformalize` on a separate 57-row partition, then `predict_set` on unseen
measurements. Sets can be benign-only, malignant-only, both classes or empty.
The library supplies the finite-sample conformity-score quantile. Under
exchangeability its coverage statement is marginal, not patient-specific,
conditional, subgroup-specific or robust to arbitrary distribution shift.

## Uncertainty and feature-space screening

Binary predictive entropy uses base-2 logarithms and lies between 0 and 1 bit.
Confidence is `max(p_malignant, 1-p_malignant)`. Independent deep networks provide
probability standard deviation. Twenty MC Dropout passes through the first
network provide a mean and standard deviation. They describe neural behavior,
even when a classical estimator supplies the primary calibrated probability.
These neural dispersions are not calibrated probabilities or guarantees.

Stateless dropout draws use stable row-derived seeds so batch order does not
change a case's uncertainty. Only dropout activates during these passes. Other
layers remain in inference mode. This avoids BatchNorm statistic updates if the
architecture later includes such a layer. The current architecture has none.

A training-only standardizer and Ledoit–Wolf shrinkage covariance produce squared
Mahalanobis distances. OOD screening flags distances above 1.5 times the fitting
set's 99.5th percentile, or a measurement above twice its fitting maximum.
Negative/nonfinite/missing values are invalid. These are simple, deliberately
conservative screens, not a clinically validated detector. They can reject valid
historical examples and miss novel deployment shifts.

## Safety policy

The gate returns `LOW_RISK_ESTIMATE`, `HUMAN_REVIEW_REQUIRED` or
`OUT_OF_SCOPE_INPUT`. Only benign-only conformal cases can be accepted. Invalid
signals and inputs fail closed. OOD inputs are out of scope. Malignant
probability at or above the acceptance threshold requires review. Entropy above
0.65 bits, ensemble standard deviation above 0.15, or MC standard deviation above
0.15 requires review. All fixed limits are prespecified in the YAML config.

On the 68 policy-development rows, a bounded sweep from 0.01 to 0.49 chooses the
threshold accepting the most cases with **zero observed accepted malignancies**
and at least 20% coverage (equivalently, at most 80% review/rejection). Ties choose
the smaller threshold. If no candidate qualifies, the gate accepts no cases.
This empirical constraint is not a statistical guarantee. The policy sample is
small and reuse for threshold selection makes its observed risk optimistic.
The held-out test estimates the complete frozen procedure.

## Evaluation definitions

Ordinary classifier metrics use threshold 0.5 with malignant as the positive
class. Accuracy and recall intervals use Wilson's 95% interval. AUC intervals
use 1,000 seeded stratified bootstrap resamples of cached test predictions.
These intervals condition on this split/training run; they do not quantify all
model-selection uncertainty. ECE uses ten equal-width malignant-probability
bins, weighting absolute observed-frequency discrepancies by bin size.
PR-AUC means average precision, not trapezoidal PR integration.

Automated coverage is the number of low-risk estimates divided by all cases.
Review rate includes ordinary referral **and out-of-scope rejection**. All
accepted decisions estimate benign, so accepted accuracy is the benign fraction
among accepted cases. Accepted-case false-negative rate uses accepted cases as
its denominator. Malignant escape rate instead uses all true malignant cases.
Malignant referral rate is its complement. Undefined accepted accuracy remains
null when coverage is zero. Conformal coverage and set size are reported
separately from gate coverage.

Confidence-ranked risk curves assess selective ordinary-classifier errors. Gate
threshold sweeps assess low-risk acceptance and workload. Test sweeps are
descriptive only and never update the policy. The UI's what-if slider reads
saved results; it cannot change the deployed threshold.

Controlled perturbations use policy-development rows: Gaussian measurement
noise, a +10% radius shift, missing texture, negative radius and a tenfold scale
change. They are stress tests, not external clinical OOD validation. Ablations
compare ordinary classification, calibration, uncertainty-based acceptance and
the complete gate. They describe component changes rather than causal effects.

## Explanations and human review

SHAP permutation explanations target the calibrated malignant probability with
24 fitting-background rows and a bounded 183 evaluations per explained case.
Global SHAP averages eight development examples. Complementary permutation
importance uses eight repeats and balanced accuracy on all 68 policy rows.
Positive local SHAP values move probability toward malignant relative to the
background. Correlated inputs may share or redistribute attribution, and
marginal feature permutations may create implausible combinations. Neither
method establishes causality. SHAP's local contributions should sum to the
predicted probability minus its baseline, within numerical precision.

SQLite stores anonymous case IDs, measurements, original JSON predictions,
model version, reviewer decisions and notes. Triggers prevent updates to the
original record and updates/deletes of audit rows through ordinary SQL. A local
database owner can still alter the schema, so this is not tamper-proof storage.
Exports exclude notes and measurements. Review never triggers automatic
retraining. This local teaching application has no user authentication and
should not store real patient information.

## Reproducibility and limitations

Configuration, seeds, partition indices, data digest, dependency versions,
model hashes, histories and predictions accompany each run. Completed runs
cannot be silently overwritten. `evaluate.py` reads cached predictions and
metrics without another test prediction pass. Native Keras exports and a trusted
joblib bundle preserve the models. Loading rejects missing, changed or
incompatible components before inference. Hashes detect accidental mismatch,
not malicious replacement of both files and manifest.

The experiment uses a historical, small, single-source dataset. Strong feature
correlations, limited calibration/conformal sample sizes, unavailable patient
linkage, unknown deployment distribution and absence of prospective external
validation limit every claim. High held-out accuracy does not establish
clinical utility. A low-risk estimate is never clinical clearance.

## References

1. Wolberg et al. (1995), [WDBC dataset and provenance](https://doi.org/10.24432/C5DW2B), UCI.
2. Street, Wolberg and Mangasarian (1993), *Nuclear feature extraction for breast tumor diagnosis*, [DOI](https://doi.org/10.1117/12.148698).
3. Guo et al. (2017), *On Calibration of Modern Neural Networks*, [PMLR](https://proceedings.mlr.press/v70/guo17a.html).
4. Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, [paper](https://arxiv.org/abs/1612.01474).
5. Gal and Ghahramani (2016), *Dropout as a Bayesian Approximation*, [PMLR](https://proceedings.mlr.press/v48/gal16.html).
6. Angelopoulos and Bates (2021), *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, [paper](https://arxiv.org/abs/2107.07511).
7. Lundberg and Lee (2017), *A Unified Approach to Interpreting Model Predictions*, [paper](https://arxiv.org/abs/1705.07874).
8. [MAPIE documentation](https://mapie.readthedocs.io/en/stable/) and [reuse audit](third_party_reuse.md).
