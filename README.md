    # Phish or Legit?

Binary phishing vs legitimate message classification with TF-IDF + logistic regression.

## Dataset

[LHRS-UM-FERI/MENTHOS-dataset-spam](https://huggingface.co/datasets/LHRS-UM-FERI/MENTHOS-dataset-spam)
(real emails + SMS, downloaded automatically on first run).

## Run

    pip install -r requirements.txt
    streamlit run app.py

## Evaluation

    python -m src.evaluate

## Method

- Train / validation / test split (stratified).
- Model fitted on train only.
- Decision threshold selected on validation with the policy "recall >= 0.90".
- Metrics and confusion matrix reported once on the test set at the policy threshold.
- Per-feature contribution to the log-odds for logistic regression.

## Notes

- The tool supports a human decision and does not block messages automatically.
- Performance on live traffic may be lower than on the test set.