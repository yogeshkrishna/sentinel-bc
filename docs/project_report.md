# SENTINEL-BC: technical report

**Uncertainty-Aware Breast Mass Classification with Calibrated Predictions, a Safety Gate, and Human Review**
Samsung Innovation Campus — Artificial Intelligence
RV College of Engineering, Bengaluru
Problem Statement 8 · Experiment `20261106T070142Z-full`

## 1. Abstract

SENTINEL-BC evaluates selective classification on numerical WDBC morphology
measurements. Four classical families and fresh Keras networks compete using
development cross-validation. A held-out sigmoid calibrator, independent MAPIE
prediction sets, neural uncertainty signals and an explicit safety gate support
human referral. On the locked 114-row test, the development-selected SVM achieved
95.61% accuracy (109/114) and 90.48%
malignant recall, with four false negatives. The complete gate accepted 38 cases
with no observed accepted malignancies and referred every malignant test case.
This occurred at only 33.33% automated coverage. Small samples,
historical data and absent external validation prevent clinical conclusions.

## 2. Introduction

A high-confidence label gives no direct estimate of reliability when data are
ambiguous or unfamiliar. A usable research prototype must expose probabilities,
uncertainty and a measurable referral policy. WDBC provides a compact setting
where these components can be trained on an ordinary CPU and explained to a
faculty audience. Its task is classifying extracted mass measurements. Raw-image
diagnosis and patient treatment decisions fall outside the project scope.

## 3. Problem statement

Given 30 WDBC measurements, estimate malignant probability and decide whether
there is sufficient support for a low-risk research estimate. Refer potential
malignancies and ambiguous or uncertain cases. Reject invalid or strongly
unfamiliar inputs. Preserve model output if a human reviewer later disagrees.
The central question concerns the reliability/workload trade-off rather than
accuracy alone. The design must avoid leakage, class inversion and selection
based on the final test.

## 4. Literature survey

Street, Wolberg and Mangasarian describe extracting nucleus morphology for this
task. Guo et al. examine neural-network calibration and post-hoc scaling.
Lakshminarayanan et al. use independent neural ensembles to estimate predictive
uncertainty. Gal and Ghahramani connect dropout to approximate Bayesian inference.
Angelopoulos and Bates explain conformal sets and the distinction between
marginal and conditional coverage. Lundberg and Lee provide the SHAP attribution
framework. These works motivate components; they do not establish this
prototype's clinical validity. Source links appear in section 19.

## 5. Existing systems

TrustBreast implements ensemble, uncertainty and explanation experiments, but
some research notebooks depend on locked models and notebook state. MAPIE
provides a maintained conformal dependency. SHAP provides reusable attribution.
Keras Uncertainty's stochastic wrappers add unnecessary compatibility and LGPL
obligations for this compact project. Two smaller classification/selective
prediction repositories provide methodological context but expose no verified
code license. We copied none of their source or weights. The reuse audit records
the exact inspected notebook/source files and revisions.

## 6. Research gap

An accuracy-centered notebook often leaves abstention rules, class ordering,
calibration data roles, artifact coherence and review history implicit. This
project brings those contracts into a single testable application. Its
contribution is engineering integration and explicit policy evaluation, not a
new learning or conformal algorithm and not the first application of abstention
to breast mass classification.

## 7. Objectives

Train every prediction model from scratch; compare classical and neural models;
maintain a locked test partition; expose calibrated probabilities and multiple
uncertainty signals; use independent conformalization; implement a monotone
referral gate; preserve human review history; provide genuine reproducible
results and a local interface. A 95% accuracy target is an aspiration, never a
reason to alter seeds or tune against the test set.

## 8. Dataset and preprocessing

The scikit-learn equivalent of UCI WDBC contains 569 rows: 357 benign and 212
malignant. All 30 numerical features describe nucleus morphology. Identifiers
are excluded. We remap sklearn's original labels to BENIGN=0, MALIGNANT=1 before
any split. Exact duplicate rows trigger an error; patient linkage is unavailable,
so related observations cannot be excluded beyond exact duplicates. The fitting
data have no missing values, but fold-local median imputation supports a stable
training transform. Inference instead rejects missing measurements.

The fixed stratified allocation is 273 fitting, 57 probability calibration, 57
conformalization, 68 safety development and 114 locked test rows. Imputation and
scaling fit within the relevant training fold. Neural early stopping reserves
18% of each current training partition, with its scaler fitted on the inner
training rows. Fitting-only covariance estimates support OOD screening. No
SMOTE, synthetic patient generation or incompatible secondary dataset is used.
Detailed fitting distributions and correlations are saved in `data_audit.json`.

## 9. Model development

Logistic regression and SVM search C=0.3 and C=3.0. Random forest uses 200 trees
and depth 3/unrestricted; XGBoost uses 150 trees, learning rate 0.05 and depth
2/3. The MLP uses 64/32 ReLU units, L2=0.001, dropout=0.25/0.20 and a sigmoid
output. Adam minimizes binary cross-entropy. Early stopping restores the best
validation loss. Three final networks use seeds 42, 142 and 242. Their mean
probability forms the deep ensemble. Histories are actual training outputs.

Three fitting-only stratified CV folds compare mean accuracy and malignant
recall, including neural candidates. Their equal-weight mean selects the primary
model, with NLL as a tie-breaker. The SVM wins this prespecified development rule.
Its setting is C=3.0. Candidate test scores are descriptive and cannot replace
that selection after the fact.

## 10. Calibration and uncertainty estimation

A FrozenEstimator keeps the selected base weights unchanged while sklearn fits
sigmoid calibration on 57 separate rows. Test Brier changes from
0.0266 to 0.0324, NLL from
0.0840 to 0.1224, and ECE from
0.0368 to 0.0647. These are worse held-out
values. Small calibration samples can produce noisy post-hoc mappings; the
project reports the deterioration instead of claiming universal improvement.

Entropy and confidence derive from the calibrated primary probability. The
three-network standard deviation and 20 MC Dropout passes through the first
network provide separate neural signals. Stateless row-derived seeds make MC
estimates consistent across batching. Non-dropout layers stay in inference mode.
No BatchNorm layer is present. These signals do not constitute calibrated
individual risk bounds. Ledoit–Wolf Mahalanobis distance and broad training-only
range screens flag unfamiliar feature vectors.

## 11. Conformal prediction

MAPIE 1.2.0 uses prefit split conformal classification with LAC scores at 90%
nominal confidence. Only 57 independent conformalization rows determine its
scores. Test coverage is 94.74%, with mean set size
0.982, 2 empty sets and 0
both-class sets. Malignant-only subgroup coverage is
88.10%, demonstrating that marginal coverage does not
imply nominal subgroup coverage. Empty and non-benign-only sets require review.
Exchangeability is an assumption, not a deployment fact.

## 12. Safety-gate methodology

The standalone gate accepts only valid, in-scope, benign-only conformal cases
with malignant probability below 0.02,
entropy at most 0.65 bits, ensemble standard deviation at most 0.15 and MC
standard deviation at most 0.15. A 49-point probability sweep maximizes acceptance
on 68 development rows subject to zero observed accepted malignancies and at
least 20% coverage. A failed constraint search defaults to accepting nobody.
Probability monotonicity and malignant referral are tested directly.

Development accepted 20/68 cases. On the locked test the gate accepted
38/114, referred 42 malignant and
34 benign cases, and rejected 0 as OOD.
Accepted accuracy is 100.00% at **33.33%
coverage**, with a Wilson interval of 90.82%–100%.
The 0 observed accepted malignancies do not imply zero real-world risk.
Review/rejection workload is 66.67%. A low-risk research estimate
is never clinical clearance.

## 13. System architecture

Training, calibration, conformalization, uncertainty, safety, inference,
explanation, evaluation, persistence and UI have separate modules. A manifest
binds feature order, canonical classes, runtime versions and SHA-256 digests to
one model version. The inference engine fails on missing or altered components.
The Gradio app has Overview, Prediction Console, Model Comparison, Reliability
and Safety, Human Review Queue, and Research and Methodology tabs.

SQLite records anonymous cases and original predictions separately from human
decisions. Atomic transactions append review audit records. Triggers prevent
ordinary modifications to the original model record and prior audit entries.
Exports omit notes and measurements. No reviewer annotation triggers training.
The app binds only to localhost and has no authentication or clinical record
system capabilities. See [architecture](architecture.md).

## 14. Experimental results

| Model | CV accuracy (mean ± SD) | CV malignant recall | Test accuracy | Test malignant recall | FN | Brier |
|---|---:|---:|---:|---:|---:|---:|
| logistic | 98.17% ± 0.63 pp | 97.09% | 96.49% | 90.48% | 4 | 0.0204 |
| rbf_svm | 97.80% ± 2.91 pp | 98.04% | 95.61% | 90.48% | 4 | 0.0266 |
| random_forest | 95.97% ± 0.63 pp | 96.13% | 95.61% | 88.10% | 5 | 0.0375 |
| xgboost | 97.07% ± 1.27 pp | 97.11% | 96.49% | 90.48% | 4 | 0.0344 |
| mlp | 97.44% ± 1.68 pp | 95.13% | 98.25% | 95.24% | 2 | 0.0174 |
| deep_ensemble | 97.80% ± 0.00 pp | 96.11% | 97.37% | 95.24% | 2 | 0.0193 |

All test results use the same 114 observations. Primary calibrated accuracy is
95.61%, with Wilson 95% interval
90.14%–98.11%. Malignant recall is
90.48%, with interval
77.93%–96.23%. Precision is
97.44%, specificity 98.61%, F1 0.9383,
ROC-AUC 0.9967, and average precision 0.9945. The primary
confusion matrix is [[71,1],[4,38]], ordered benign then malignant. Bootstrap
AUC intervals and individual source-row false negatives appear in cached
metrics and CSVs. Intervals condition on this experiment and do not capture all
training or selection uncertainty.

The MLP obtains the highest descriptive test accuracy, but its development
score did not win. This discrepancy is expected in small samples. It cannot
justify replacing the chosen SVM after test inspection. Training time and
inference latency are machine-specific and are recorded with the experiment.
Deep ensemble and MC uncertainty remain real independently trained components.

## 15. Robustness and ablation studies

| Stage | Coverage | Accepted accuracy | Accepted malignant |
|---|---:|---:|---:|
| base_classifier | 65.79% | 94.67% | 4 |
| calibrated_classifier | 65.79% | 94.67% | 4 |
| calibrated_with_uncertainty | 33.33% | 100.00% | 0 |
| full_safety_gate | 33.33% | 100.00% | 0 |


For the first two stages, acceptance means a benign ordinary prediction at
threshold 0.5. Uncertainty acceptance adds the development-selected probability
threshold and prespecified uncertainty limits. Full acceptance also requires
conformal support and valid/in-scope inputs. These descriptive ablations reveal
component/workload effects and do not isolate causal effects of individual gates.

| Controlled perturbation | n | Out-of-scope fraction | Gate coverage | Accepted malignant |
|---|---:|---:|---:|---:|
| unmodified_policy_data | 68 | 1.47% | 29.41% | 0 |
| noise_005_train_std | 68 | 1.47% | 32.35% | 0 |
| noise_020_train_std | 68 | 10.29% | 20.59% | 0 |
| radius_plus_10pct | 68 | 1.47% | 27.94% | 0 |
| missing_texture | 68 | 100.00% | 0.00% | 0 |
| implausible_10x | 68 | 100.00% | 0.00% | 0 |
| negative_radius | 68 | 100.00% | 0.00% | 0 |


These controlled tests use 68 policy-development rows and fixed perturbations.
They neither estimate clinical distribution-shift performance nor provide
independent external validation. Noise can create invalid values in naturally
zero-valued morphology features, so validity checks may reject them. This is an
intentional observable limitation of the stress design. Threshold sensitivity
plots are descriptive on cached test predictions and never change deployment.

SHAP targets calibrated malignant probability with 24 fitting-background rows
and eight global development examples. Local attributions identify direction
relative to that baseline. Eight-repeat permutation importance uses balanced
accuracy on development rows. Correlated features can split attribution, and
permutation can create implausible combinations. Neither importance is causal.

## 16. Limitations and responsible AI

The dataset is small, historical and single-source. Calibration/conformal/policy
subsets are necessarily small, and preserving valid roles reduces fitting data.
Patient linkage and consistent physical units are unavailable. Feature
correlations complicate attributions. Distribution shift, changing acquisition
processes and demographic/subgroup reliability remain unvalidated. The gate
increases review workload and cannot guarantee safety. Clinical deployment,
prospective studies, medical-device compliance and patient treatment fall
outside this project. Human notes and local databases stay out of Git.

## 17. Conclusion

The executed project meets the 95% accuracy aspiration on its frozen primary
test while exposing four ordinary false negatives. Its referral policy catches
every malignant test case at one-third automated coverage. Calibration worsens
this split's proper scoring rules. Reporting all three observations together
supports a more credible presentation than quoting selective accuracy alone.
The result is an independently retrainable academic integration.

## 18. Future work

Acquire an independently governed external cohort with comparable measurement
definitions, evaluate prospective workload and subgroup behavior, increase
calibration/policy sample sizes, and compare cross-fitted calibration under a
preregistered protocol. Any replacement model requires new development
selection and a genuinely new final evaluation cohort. Clinical workflow and
record controls require separate domain-led design and validation.

## 19. Verified references and attribution

The references in [methodology](methodology.md#references) link to UCI, publisher
or author paper pages. The [reuse audit](third_party_reuse.md) records actual
source inspection of the six required repositories, exact revisions and license
decisions. We use established libraries and claim original integration/policy
engineering only. Project source is MIT; WDBC is CC BY 4.0. No pretrained
classification weights or upstream experiment results enter this run.

## Execution evidence

Full and quick experiments completed on Windows/Python 3.12.10. Local tests
cover genuine classical/Keras training, label/feature order, split separation,
calibration, MAPIE, reload, safety monotonicity, CSV consistency, corrupt/missing
artifacts, SQLite persistence/audit and Gradio construction. Browser smoke tests
exercise a labeled prediction and human review. Final verification details,
reproducibility comparison and CI link are recorded in `verification.json` next
to the metrics. Cross-platform full-training bitwise equivalence is not claimed.
