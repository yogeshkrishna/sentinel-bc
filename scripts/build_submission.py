"""Refresh written submission materials from the saved full experiment.

Run from an installed checkout: python scripts/build_submission.py
This command reads cached results; it never fits models or evaluates the test set.
"""
import json
from pathlib import Path
import nbformat as nb
from sentinel_bc.config import ROOT


def pct(v):
    return "undefined" if v is None else f"{100*v:.2f}%"


def build():
    directory = ROOT / "results/full"
    metrics = json.loads((directory / "metrics.json").read_text())
    cv = json.loads((directory / "cross_validation.json").read_text())
    ablation = json.loads((directory / "ablation.json").read_text())
    stress = json.loads((directory / "robustness.json").read_text())
    m, s, c = metrics["test"], metrics["selective"], metrics["conformal"]
    best_cv = {}
    for r in cv:
        if r["model"] not in best_cv or r["selection_score"] > best_cv[r["model"]]["selection_score"]:
            best_cv[r["model"]] = r
    header = "| Model | CV accuracy (mean ± SD) | CV malignant recall | Test accuracy | Test malignant recall | FN | Brier |\n|---|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for name, result in metrics["models"].items():
        r = best_cv[name]
        rows.append(f"| {name} | {pct(r['accuracy_mean'])} ± {100*r['accuracy_std']:.2f} pp | {pct(r['malignant_recall_mean'])} | {pct(result['accuracy'])} | {pct(result['malignant_recall'])} | {result['fn']} | {result['brier']:.4f} |")
    table = header + "\n".join(rows)
    ablation_table = "| Stage | Coverage | Accepted accuracy | Accepted malignant |\n|---|---:|---:|---:|\n"
    for name, result in ablation.items():
        a = result["selective"]
        ablation_table += f"| {name} | {pct(a['coverage'])} | {pct(a['accepted_accuracy'])} | {a['accepted_malignant']} |\n"
    stress_table = "| Controlled perturbation | n | Out-of-scope fraction | Gate coverage | Accepted malignant |\n|---|---:|---:|---:|---:|\n"
    for r in stress:
        stress_table += f"| {r['experiment']} | {r['n']} | {pct(r['selective']['out_of_scope']/r['n'])} | {pct(r['selective']['coverage'])} | {r['selective']['accepted_malignant']} |\n"
    readme = f'''# SENTINEL-BC

### Uncertainty-aware breast mass classification with a safety gate and human review

A local research console for exploring when a classifier should defer. SENTINEL-BC
combines classical machine learning and fresh Keras networks with calibrated
probabilities, MAPIE prediction sets, neural uncertainty signals and an explicit
referral policy. Every reported result comes from this repository's executed pipeline.

**Samsung Innovation Campus — Artificial Intelligence**
RV College of Engineering, Bengaluru · Problem Statement 8

**Research only.** This is an academic prototype for extracted numerical WDBC
measurements. It does not diagnose images, establish clinical utility or provide
clinical clearance. Do not enter real patient data.

![The running research console](docs/images/overview.png)

## What works

- Four classical model families, a TensorFlow/Keras MLP and three independently trained ensemble members.
- Fold-local preprocessing, fixed development/test roles and a single locked final evaluation.
- Held-out sigmoid calibration, predictive entropy, deep ensemble disagreement and repeatable MC Dropout.
- Independent 90% MAPIE split conformal sets, training-only OOD screening and a tested safety gate.
- Six Gradio tabs, 30-feature input validation, bounded CSV inference, SHAP explanations and downloadable reports.
- SQLite human review, immutable original predictions, append-only audit records and anonymous exports.

## Executed results

Version `{metrics['version']}`. The final test contains **114 observations: 72 benign
and 42 malignant**. Positive class is always **MALIGNANT = 1**.
All candidate models and safety thresholds froze before the test evaluation.

The development-selected **RBF SVM** achieved **{m['correct']}/{m['n']} correct
({pct(m['accuracy'])})**, with a Wilson 95% accuracy interval of
**{pct(m['accuracy_ci95'][0])}–{pct(m['accuracy_ci95'][1])}**. Malignant recall was
**{pct(m['malignant_recall'])} ({m['tp']}/42)**, with **{m['fn']} false negatives**
and **{m['fp']} false positive**. ROC-AUC was {m['roc_auc']:.4f}.

{table}

The single MLP performed better on this test, but the SVM retains deployment
selection because development results chose it. Switching after test inspection
would contaminate model selection. Candidate test metrics above are descriptive.

The complete safety gate accepted **{s['accepted']}/114 ({pct(s['coverage'])})**
as low-risk estimates and referred **76/114 ({pct(s['review_rate'])})**.
Accepted accuracy was **{pct(s['accepted_accuracy'])}**, with a 95% Wilson interval
of **{pct(s['accepted_accuracy_ci95'][0])}–100%**. It accepted **0 observed malignant
cases** and referred **42/42 malignant cases**. These finite-sample observations
do not establish zero future risk. Prediction-set coverage was
**{pct(c['empirical_coverage'])}**, with mean set size **{c['mean_set_size']:.3f}**.

Calibration did **not** improve this held-out split: Brier score changed from
{metrics['uncalibrated']['brier']:.4f} to {m['brier']:.4f}, and NLL from
{metrics['uncalibrated']['nll']:.4f} to {m['nll']:.4f}. We retain the prespecified
calibration protocol and disclose the result.

## Start on Windows

Install **64-bit Python 3.12** from Python's official installer, including the
`py` launcher. Open PowerShell in the checkout:

```powershell
.\\setup.bat
.\\.venv\\Scripts\\python train.py --mode quick
.\\.venv\\Scripts\\python train.py --mode full
.\\.venv\\Scripts\\python -m pytest -q
.\\run.bat
```

Open [the local application](http://127.0.0.1:7860). Stop it with Ctrl+C.
CPU operation requires no account, API key, GPU or paid service. Native Windows
TensorFlow uses the CPU. GPU support depends on a compatible TensorFlow/WSL2 or
Linux installation and is optional. This exact build ran on Windows/Python 3.12.10.

Models are already trained in the delivered local workspace. A fresh clone can
train using the commands above, or download **trained-full.zip** from the
[v1.0.0 release](https://github.com/yogeshkrishna/sentinel-bc/releases/tag/v1.0.0)
and extract it at the repository root after setup. That archive contains only
our own trained artifacts and their matching results. The source repository
excludes executable model serialization and runtime review records.

`requirements.txt` pins direct dependencies. `requirements-lock.txt` records the
tested complete Windows resolution. `setup.bat` installs the latter and then the
package. On Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip check
python train.py --mode full
python app.py
```

Linux CI tests installation and small real-training integrations. A full Linux
experiment has not been executed. Version/hash checks protect local artifacts
from accidental mixing. Load only trusted joblib bundles; hashes are not a
security signature.

## Training and reproducibility

```powershell
# Full experiment; refuses to replace an existing local trained run
.\\.venv\\Scripts\\python train.py --mode full

# A separate repeat, preserving the completed experiment
.\\.venv\\Scripts\\python train.py --mode full --output-root experiments/repeat

# Read cached test metrics and regenerate plots without predicting again
.\\.venv\\Scripts\\python evaluate.py --figures

# Refresh the written report and notebook from cached results
.\\.venv\\Scripts\\python scripts/build_submission.py
```

Quick mode trains logistic regression, SVM and one smaller Keras experiment.
Full mode trains all six candidates, including a three-member deep ensemble.
They use separate output directories. Full mode completed here, along with quick
mode and a separate quick reproducibility check. The evaluator never retunes to
the final test. Neural early stopping uses an inner validation partition.
Reproducibility evidence and executed checks appear in the technical report.

`app.py` never trains. Missing or incompatible artifacts produce an explicit
setup screen. Inference works offline after installation/training. Completed
local runs are protected by the artifact manifest. Published result files in a
fresh clone are reference evidence and are replaced by the first genuine local
training run. Use a separate output root to preserve another experiment.

## Data and safety policy

[UCI WDBC, ID 17](https://doi.org/10.24432/C5DW2B), CC BY 4.0, supplies 569 rows and
30 morphology features through scikit-learn's equivalent bundle. The loader
explicitly inverts sklearn's label ordering to `0=BENIGN`, `1=MALIGNANT`. Source
units are not consistently specified, so no physical-unit conversion is assumed.

| Role | Rows |
|---|---:|
| Fitting / 3-fold CV | 273 |
| Sigmoid calibration | 57 |
| Independent MAPIE conformalization | 57 |
| Safety policy development | 68 |
| Locked stratified test | 114 |

The gate requires valid, in-scope measurements, a benign-only conformal set,
malignant probability below **{metrics['policy']['max_p_malignant']:.2f}**, and
prespecified entropy/dispersion limits. A 49-point development sweep maximizes
coverage with zero observed accepted malignancies and at least 20% coverage.
If infeasible, all cases require review. Increasing malignant probability can
never turn a referral into acceptance while other signals stay fixed.

```mermaid
flowchart LR
    A[30 numerical measurements] --> B[Validity and OOD screen]
    A --> C[Calibrated probability]
    A --> D[Neural uncertainty]
    C --> E[MAPIE prediction set]
    B --> F[Frozen safety gate]
    C --> F
    D --> F
    E --> F
    F --> G[Low-risk research estimate]
    F --> H[Human review / out of scope]
    H --> I[SQLite review and audit]
```

See [methodology](docs/methodology.md) for denominators, uncertainty definitions,
OOD assumptions and statistical limits, and [architecture](docs/architecture.md)
for module boundaries. The UI's threshold slider is educational and cannot
change the deployed policy. Human annotations never trigger retraining.

## Evidence and submission materials

| Material | Location |
|---|---|
| Academic technical report, literature survey and evaluation | [Project report](docs/project_report.md) |
| Twelve-slide editable PowerPoint | [Presentation](docs/presentation.pptx) |
| Educational runnable notebook | [Experiments](notebooks/experiments.ipynb) |
| Five-minute demonstration | [Demo script](docs/demo_script.md) |
| Saved predictions, metrics, ablations and stress tests | [Full experiment](results/full/) |
| ROC/PR, calibration, safety, neural learning and importance figures | [Figures](results/full/figures/) |
| References, inspected revisions and license decisions | [Reuse audit](docs/third_party_reuse.md) |

## Limitations

WDBC is small and historical, with correlated features and unavailable patient
linkage. Calibration and policy estimates are noisy. There is no prospective or
independent clinical validation. Conformal guarantees require exchangeability
and are marginal. Synthetic perturbations do not establish real-world OOD
performance. The gate's conservative coverage creates substantial review work.
The application lacks clinical authentication, workflow validation and regulated
record management. Use anonymous teaching examples only.

## Attribution and team

MIT-licensed project source. Data and dependencies retain their own licenses.
We use scikit-learn, TensorFlow/Keras, XGBoost, MAPIE, SHAP and Gradio rather than
reimplementing their algorithms. The [reuse audit](docs/third_party_reuse.md)
records all six requested references. No third-party trained weights or source
snippets were copied. Models in the release were trained by this pipeline.

| Team member | Contribution to document before submission |
|---|---|
| Name / USN | Data and experiment ownership |
| Name / USN | Reliability methods and safety policy |
| Name / USN | Application, evaluation and presentation |

Replace these explicit placeholders with the team's actual work. Do not claim
unperformed contributions or clinical validation.
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    report = f'''# SENTINEL-BC: technical report

**Uncertainty-Aware Breast Mass Classification with Calibrated Predictions, a Safety Gate, and Human Review**
Samsung Innovation Campus — Artificial Intelligence
RV College of Engineering, Bengaluru
Problem Statement 8 · Experiment `{metrics['version']}`

## 1. Abstract

SENTINEL-BC evaluates selective classification on numerical WDBC morphology
measurements. Four classical families and fresh Keras networks compete using
development cross-validation. A held-out sigmoid calibrator, independent MAPIE
prediction sets, neural uncertainty signals and an explicit safety gate support
human referral. On the locked 114-row test, the development-selected SVM achieved
{pct(m['accuracy'])} accuracy ({m['correct']}/114) and {pct(m['malignant_recall'])}
malignant recall, with four false negatives. The complete gate accepted 38 cases
with no observed accepted malignancies and referred every malignant test case.
This occurred at only {pct(s['coverage'])} automated coverage. Small samples,
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
{metrics['uncalibrated']['brier']:.4f} to {m['brier']:.4f}, NLL from
{metrics['uncalibrated']['nll']:.4f} to {m['nll']:.4f}, and ECE from
{metrics['uncalibrated']['ece']:.4f} to {m['ece']:.4f}. These are worse held-out
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
scores. Test coverage is {pct(c['empirical_coverage'])}, with mean set size
{c['mean_set_size']:.3f}, {c['empty_sets']} empty sets and {c['both_class_sets']}
both-class sets. Malignant-only subgroup coverage is
{pct(c['malignant_coverage'])}, demonstrating that marginal coverage does not
imply nominal subgroup coverage. Empty and non-benign-only sets require review.
Exchangeability is an assumption, not a deployment fact.

## 12. Safety-gate methodology

The standalone gate accepts only valid, in-scope, benign-only conformal cases
with malignant probability below {metrics['policy']['max_p_malignant']:.2f},
entropy at most 0.65 bits, ensemble standard deviation at most 0.15 and MC
standard deviation at most 0.15. A 49-point probability sweep maximizes acceptance
on 68 development rows subject to zero observed accepted malignancies and at
least 20% coverage. A failed constraint search defaults to accepting nobody.
Probability monotonicity and malignant referral are tested directly.

Development accepted 20/68 cases. On the locked test the gate accepted
{s['accepted']}/114, referred {s['referred_malignant']} malignant and
{s['referred_benign']} benign cases, and rejected {s['out_of_scope']} as OOD.
Accepted accuracy is {pct(s['accepted_accuracy'])} at **{pct(s['coverage'])}
coverage**, with a Wilson interval of {pct(s['accepted_accuracy_ci95'][0])}–100%.
The 0 observed accepted malignancies do not imply zero real-world risk.
Review/rejection workload is {pct(s['review_rate'])}. A low-risk research estimate
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

{table}

All test results use the same 114 observations. Primary calibrated accuracy is
{pct(m['accuracy'])}, with Wilson 95% interval
{pct(m['accuracy_ci95'][0])}–{pct(m['accuracy_ci95'][1])}. Malignant recall is
{pct(m['malignant_recall'])}, with interval
{pct(m['recall_ci95'][0])}–{pct(m['recall_ci95'][1])}. Precision is
{pct(m['precision'])}, specificity {pct(m['specificity'])}, F1 {m['f1']:.4f},
ROC-AUC {m['roc_auc']:.4f}, and average precision {m['pr_auc']:.4f}. The primary
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

{ablation_table}

For the first two stages, acceptance means a benign ordinary prediction at
threshold 0.5. Uncertainty acceptance adds the development-selected probability
threshold and prespecified uncertainty limits. Full acceptance also requires
conformal support and valid/in-scope inputs. These descriptive ablations reveal
component/workload effects and do not isolate causal effects of individual gates.

{stress_table}

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
'''
    (ROOT / "docs/project_report.md").write_text(report, encoding="utf-8")
    notebook = nb.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3 (SENTINEL-BC)", "language": "python", "name": "python3"}
    notebook.cells = [
        nb.v4.new_markdown_cell("# SENTINEL-BC experiments\n\nThis notebook uses the application's modules and saved experiment. It never builds a second pipeline or silently refits. Run `python train.py --mode full` first. Research measurements only, never clinical clearance."),
        nb.v4.new_code_cell("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nfrom sentinel_bc.config import ROOT, load_config\nfrom sentinel_bc.data import load_data, partitions\nfrom sentinel_bc.inference import Engine\nfrom sentinel_bc.explainability import local_table\nfrom sentinel_bc.safety_gate import SafetyPolicy, decide\nx, y = load_data()\nroles = partitions(y, load_config())\nassert sum(y == 1) == 212\nassert len(np.unique(np.concatenate(list(roles.values())))) == 569\npd.DataFrame([{\"role\": k, \"n\": len(v), \"malignant\": int(y[v].sum())} for k, v in roles.items()])"),
        nb.v4.new_markdown_cell("## Fitting-only data inspection\nCorrelated morphology measurements can share attribution. Units follow the source values; no invented physical-unit conversion is applied."),
        nb.v4.new_code_cell("x.iloc[roles['fit']].describe().round(3)"),
        nb.v4.new_markdown_cell("## The frozen experiment\nCV selected the primary estimator before any final test metrics. The table below reads cached test results."),
        nb.v4.new_code_cell("directory = ROOT / 'results/full'\nmetrics = json.loads((directory / 'metrics.json').read_text())\ndisplay(pd.DataFrame(metrics['models']).T[['accuracy', 'malignant_recall', 'fn', 'brier']])\nmetrics['selective']"),
        nb.v4.new_code_cell("display(Image(filename=str(directory / 'figures/calibration.png')))\ndisplay(Image(filename=str(directory / 'figures/safety.png')))"),
        nb.v4.new_markdown_cell("## Labeled development example\nInference uses the same version-checked engine as Gradio. The source diagnosis is educational context and is never an input feature."),
        nb.v4.new_code_cell("engine = Engine()\nexamples = pd.read_csv(directory / 'demo_cases.csv')\ncase = examples.loc[:0, engine.features]\nprint(examples.loc[0, ['example_name', 'source_label', 'source_row']].to_dict())\nprediction = engine.predict(case)[0]\nprediction"),
        nb.v4.new_code_cell("display(local_table(engine, case).head(10))"),
        nb.v4.new_markdown_cell("## Fail-closed behavior and monotonicity\nAn impossible negative radius is out of scope. Holding support and uncertainty fixed, increasing malignant probability cannot relax referral."),
        nb.v4.new_code_cell("invalid = case.copy()\ninvalid.iloc[0, 0] = -1\nassert engine.predict(invalid)[0]['decision'] == 'OUT_OF_SCOPE_INPUT'\nu = {'entropy': .1, 'disagreement': .01, 'mc_std': .01}\nflags = [decide(float(p), u, [True, False], True, False, engine.policy)[0] != 'LOW_RISK_ESTIMATE' for p in np.linspace(0,1,101)]\nassert np.all(np.diff(np.asarray(flags, dtype=int)) >= 0)\nprint('Invalid input rejected; monotone referral verified.')"),
        nb.v4.new_markdown_cell("## Retraining\nUse `python train.py --mode quick` for an installation check and `python train.py --mode full --output-root experiments/repeat` for a separate repeat. Do not tune using the locked test. A new scientific comparison needs a new evaluation plan.\n\n## Interpretation\nAlways report accepted accuracy **with coverage**. Zero observed accepted malignancies in a small sample does not establish zero real-world risk. Calibration worsened the current held-out Brier/NLL. Conformal coverage is marginal under exchangeability, not individual clinical protection.")]
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nb.write(notebook, ROOT / "notebooks/experiments.ipynb")
    print("Refreshed README, academic report and runnable notebook from cached results")


if __name__ == "__main__":
    build()
