# Research and reuse audit

Source review: 6 November 2026. No third-party model weights, checkpoints,
notebook outputs or source code have been incorporated. The application is an
original integration of the dependencies below. Dependency distributions retain
their own notices. The repository's MIT license covers our source only.

| Component | Source and inspected revision | Decision | Reason | License | Attribution |
|---|---|---|---|---|---|
| TrustBreast | [Repository](https://github.com/Iqra672-ai/TrustBreast/tree/d63a1e9ca23bd3cd8c527781f65cae2e9a2083b0), `00_train_locked_model_OPTIONAL`, `01_objective1_ensemble`, `03_objective3_uncertainty`, `07_crossconformal_out_of_fold` code cells | Methodological reference | Studied reseeding, dense/dropout networks, uncertainty and coverage evaluation. Locked-model paths and notebook execution state are unsuitable for a retrainable package. No functions copied. | MIT, copyright 2026 Iqra Mushtaq | Cite as a research implementation, not evidence for our performance. |
| MAPIE | [Repository](https://github.com/scikit-learn-contrib/MAPIE), source `mapie/classification.py`, reviewed main `c073876f271d28a05e4c2b53639293ff3a805203` | Dependency, pinned 1.2.0 | Supported split conformal classifier with `prefit=True`, separate `conformalize`, then `predict_set`. Avoid manual quantiles. | BSD-3-Clause | MAPIE contributors; [documentation](https://mapie.readthedocs.io/en/stable/). |
| SHAP | [Source](https://github.com/shap/shap/blob/v0.49.1/shap/explainers/_permutation.py), v0.49.1 and LICENSE | Dependency | Model-agnostic permutation explainer supports the calibrated probability directly. Small background and bounded evaluation budget. | MIT | SHAP authors; Lundberg and Lee (2017). |
| Keras Uncertainty | [Source](https://github.com/mvaldenegro/keras-uncertainty/tree/4269ad3a043485273582bdf23b7ad8d82b41e216), `models/stochastic_model.py`, `setup.py`, LICENSE | Reference only; no dependency or code reuse | Its stochastic wrapper and old backend assumptions add unnecessary compatibility and license obligations. Our small stateless-dropout implementation keeps non-dropout layers in inference mode. | LGPL-3.0 | Matias Valdenegro and contributors. No LGPL code incorporated. |
| BreastCancerDiagnosis_using-machine-learning | [Notebook](https://github.com/SamridhiArora/BreastCancerDiagnosis_using-machine-learning/tree/08b54c053e5cdb06db6f3ad5e643388e7c73ac18), `f1_of_Winscoin_BCAN.ipynb` | Benchmark/UI reference only | Reviewed classical comparisons, calibration and prediction functions. No explicit repository license found. Old prefit calibration patterns are replaced by sklearn FrozenEstimator on separate data. | No license detected | Samridhi Arora. No copying. |
| uncertainty-aware-ml-classifier | [Notebook](https://github.com/SiyamMukhtar/uncertainty-aware-ml-classifier/tree/ad3c1a348b4166a9b3ac3990e689d63328a06e2d), `uncertainty_m.ipynb` | Methodology reference only | Reviewed selective acceptance and calibration plots. Our safety gate has independent tests and explicit malignant referral. | No license detected | Siyam Mukhtar. No copying. |

## Other runtime dependencies

scikit-learn (BSD-3-Clause): pipelines, covariance shrinkage, cross-validation,
metrics and sigmoid calibration. TensorFlow (Apache-2.0) and Keras (Apache-2.0):
fresh neural networks and native model serialization. XGBoost (Apache-2.0):
boosted trees. Gradio (Apache-2.0): UI and local server. NumPy (BSD-3-Clause),
pandas (BSD-3-Clause), SciPy (BSD-3-Clause), matplotlib (PSF-based), joblib
(BSD-3-Clause), PyYAML (MIT), pytest (MIT), Jupyter tools (BSD-3-Clause).
SQLite is in Python's standard library. Exact tested resolutions are recorded in
`requirements-lock.txt` after installation.

## Data

[UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic),
Wolberg, Mangasarian, Street and Street (1995), DOI
[10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), CC BY 4.0. Loaded from the
equivalent scikit-learn bundle, retaining the documented feature order. Labels
are explicitly inverted from sklearn to `0=BENIGN`, `1=MALIGNANT`.

No repository benchmark claims enter our results. We did not inspect or use any
upstream trained artifact.
