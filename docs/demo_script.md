# Five-minute faculty demonstration

Use `run.bat`, then open http://127.0.0.1:7860. Full artifacts must exist. All
dropdown examples are labeled historical WDBC policy-development rows, never
real patient records. Model status and version appear in Overview.

**0:00–0:35 — Purpose and evidence.** Explain that the system classifies 30
already extracted morphology measurements. Show 109/114 primary classifier
accuracy, 4 false negatives, 38/114 gate acceptance and the research disclaimer.
State that accepted accuracy only makes sense alongside coverage.

**0:35–1:15 — Low-risk example.** Open Prediction Console and choose the labeled
low-risk benign example (WDBC row 267 in the delivered experiment). Evaluate it.
Point to calibrated malignant probability, the benign-only set and the reason
code. Explain that this is a research estimate, not clinical clearance.

**1:15–1:50 — Malignant and uncertain examples.** Select the labeled malignant
referral example, evaluate and show human review. Then select the most uncertain
development example. Show entropy and its review reason. A source label is
educational context; it never enters the model's feature vector. SHAP signs
describe attribution toward malignant/benign relative to a background, not cause.

**1:50–2:10 — Invalid / OOD.** Change mean radius to -1 and evaluate. The result
must be OUT_OF_SCOPE_INPUT. Restore the source example. A very large positive
radius also triggers the feature-space screen. Do not save impossible values
as a real case.

**2:10–3:00 — Human review.** Evaluate a valid example and click Add to review
queue. Copy its anonymous case ID. In Human Review Queue refresh, paste the ID,
inspect the case, choose FURTHER_REVIEW and enter a brief teaching note. Record
review. Show the unchanged original output and the appended HUMAN_REVIEW audit
entry. Export decisions and explain that notes and measurements are excluded.
No automatic retraining occurs.

**3:00–4:00 — Reliability dashboard.** Show calibration before/after and state
that the held-out Brier score worsened from 0.0266 to 0.0324. Explain 90% nominal
conformal prediction and 94.74% observed marginal test coverage. Show the gate's
33.33% coverage and 66.67% review rate. Move the educational threshold slider and
point out the deployed threshold remains 0.02. Synthetic noise is a controlled
stress test, not external clinical evidence.

**4:00–4:35 — Model comparison.** Show the development CV rule and the six
candidate families. The MLP scored 98.25% on the test, but the SVM remains primary
because development selected it. Show actual neural learning curves and seeds.
Changing the deployed model now based on this test would violate the protocol.

**4:35–5:00 — Reproducibility and limits.** Show the saved splits, manifest,
training command and tests. Close with historical single-source data, small
calibration/policy samples and absent prospective clinical validation. Future
work needs an independent cohort and domain-led workflow evaluation.
