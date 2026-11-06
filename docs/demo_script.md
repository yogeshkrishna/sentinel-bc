# Five-minute Training Studio demonstration

Use Python 3.12, run `setup.bat` once, then start `run.bat` and open
[Training Studio](http://127.0.0.1:7860). Use teaching measurements only.
The app keeps the historical v1.0.0 benchmark separate from student runs.

| Time | What to show |
|---|---|
| 0:00–0:45 | In **Train**, open **Preview and CSV templates** and download the labeled public WDBC example. Select **Upload labeled CSV** and upload it. Point out 569 rows, 30 features, 357 benign labels and 212 malignant labels. The app recognizes it as repeated WDBC data. |
| 0:45–1:45 | Select **Quick Training** and click **Train Models**. Show the real phase, elapsed time and recent activity. Explain that cross-validation chooses the model before calibration, conformalization, policy selection and final testing. Allow extra time on a slower CPU. |
| 1:45–2:20 | Read the completed run's test accuracy, malignant recall, false negatives and gate coverage. Click **Use This Trained Model** and confirm the active version changes. |
| 2:20–3:25 | In **Predict**, choose a labeled benign example and click **Predict**. Repeat with the malignant referral example. Compare the predicted class, probability and gate decision. Expand prediction details if asked about entropy, conformal sets or SHAP. |
| 3:25–4:15 | In **Review**, select the automatically added case by its ID. Record **FURTHER_REVIEW** and a teaching note. Expand the audit history to show the original model output and the new reviewer action. |
| 4:15–5:00 | In **Results**, report accepted accuracy together with coverage, then show the ANN curves, confusion matrix and reliability diagram. Open advanced evidence to show MAPIE coverage and move the educational threshold. The deployed threshold stays fixed. |

The public example exercises the CSV workflow on the original dataset. For a
different teaching corpus, use the template and confirm the dataset summary
before starting. Do not use a repeated WDBC run as independent external evidence.
The [historical benchmark](project_report.md) reports a separate frozen
experiment, including calibration performance and its limitations.
