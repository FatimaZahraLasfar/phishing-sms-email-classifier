# Phish or Legit? — Phishing SMS & Email Classifier

A machine learning application that detects potentially fraudulent SMS messages and emails using Natural Language Processing (NLP). It combines **TF-IDF feature extraction**, **Logistic Regression**, and an interactive **Streamlit** interface to classify messages as phishing or legitimate.

The application also provides prediction scores, interpretable feature contributions, model evaluation metrics, and a configurable decision threshold.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Evaluating the Models](#evaluating-the-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Overview

Phishing attacks often use deceptive messages to trick recipients into clicking malicious links, revealing sensitive information, or performing fraudulent actions.

**Phish or Legit?** explores how traditional machine learning and NLP can help identify suspicious messages.

Users can paste an SMS or email into the application and receive a classification, a phishing score, and explanations of the textual features that influenced the prediction.

The project emphasizes not only classification but also model evaluation, threshold selection, and interpretability.

## Features

- **Phishing detection:** Classifies messages as `PHISHING` or `LEGITIMATE`.
- **Interactive interface:** Provides a user-friendly Streamlit application.
- **TF-IDF vectorization:** Converts text into numerical features using word n-grams.
- **Logistic Regression:** Uses a supervised machine learning classifier with class weighting.
- **Configurable decision threshold:** Allows users to adjust the threshold used for classification.
- **Prediction scores:** Displays raw and calibrated phishing scores.
- **Explainable predictions:** Highlights words and features that contribute to phishing or legitimate classifications.
- **Model evaluation:** Reports precision, recall, F1-score, accuracy, Brier score, Expected Calibration Error (ECE), and ROC-AUC in the application.
- **Confusion matrix:** Helps analyze classification errors.
- **Error analysis:** Examines false positives and false negatives.
- **Automated dataset loading:** Downloads the dataset files from Hugging Face when required.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning, preprocessing, and evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | Main classification model |
| Multinomial Naive Bayes | Baseline model for comparison |
| Streamlit | Interactive web interface |
| Matplotlib | Evaluation visualizations |
| Hugging Face Hub | Dataset download |
| imbalanced-learn | Additional tools for imbalanced-data workflows |

## Dataset

The project uses the **MENTHOS Spam Dataset**, available on Hugging Face.

**Dataset:** [LHRS-UM-FERI/MENTHOS-dataset-spam](https://huggingface.co/datasets/LHRS-UM-FERI/MENTHOS-dataset-spam)

The application loads the following CSV files:

- `spam_train.csv`
- `spam_val.csv`
- `spam_test.csv`

The dataset contains message text and labels, which are transformed into a binary classification target:

- `0` — Legitimate message
- `1` — Phishing or spam message

The dataset is downloaded automatically by the data-loading pipeline, so you do not need to download the CSV files manually.

**Important:** Dataset labels and the project's binary mapping should be interpreted according to the dataset's labeling conventions.

## Project Structure

```text
phishing-sms-email-classifier/
│
├── app.py                    # Streamlit application
├── dataset_info.py           # Dataset information and inspection
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # Data loading, preprocessing,
│   │                         # model pipelines and predictions
│   └── evaluate.py           # Model evaluation and error analysis
│
├── results/                  # Generated evaluation outputs
│   ├── metrics.json
│   ├── failures.txt
│   └── phish_results.png
│
└── .gitignore
```

The `results/` directory and its files are generated when the evaluation script runs; they may not exist in a fresh checkout.

## How It Works

The classification workflow consists of the following stages.

### 1. Data Loading

The application downloads the dataset from Hugging Face and loads the message text and corresponding labels.

### 2. Text Preprocessing

The pipeline normalizes text and replaces certain patterns with standardized tokens:

- URLs are replaced with a URL token.
- Monetary amounts are replaced with a money token.
- Numbers are replaced with a numerical token.

This reduces variation between messages and helps the model learn more generalizable patterns.

### 3. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) converts messages into numerical vectors.

The main pipeline uses word-level unigrams and bigrams, allowing the classifier to learn from individual words and short word sequences.

### 4. Model Training

Logistic Regression learns the relationship between the extracted features and the target labels. Class weighting is used to account for differences between classes.

The evaluation module also supports Multinomial Naive Bayes and additional model configurations.

### 5. Decision Threshold Selection

Rather than automatically relying on a threshold of `0.5`, the main application selects a decision threshold using the validation set.

Its target policy is **recall of at least 90% for phishing messages**, with precision maximized among thresholds satisfying that target when possible.

If the target recall cannot be achieved, the selection procedure falls back to the threshold with the best F1-score on the validation set.

The final evaluation uses the separate test set.

### 6. Prediction and Explanation

When a user submits a message, the application displays the predicted class, phishing scores, and relevant feature contributions.

The feature explanations help users understand which textual patterns influenced the prediction.

## Installation

### Prerequisites

- Python installed on your computer.
- Git installed on your computer.
- An internet connection for downloading the dataset and installing dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/FatimaZahraLasfar/phishing-sms-email-classifier.git
```

### 2. Navigate to the project directory

```bash
cd phishing-sms-email-classifier
```

### 3. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can activate the environment from Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

**Linux or macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the dependencies

Upgrade pip and install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

From the project's root directory, execute:

```bash
streamlit run app.py
```

Streamlit will start a local development server and display a URL in your terminal, typically:

```text
http://localhost:8501
```

Open that address in your browser.

### Using the application

1. Open the Streamlit interface.
2. Select an example message or paste your own SMS or email.
3. Adjust the decision threshold if needed.
4. Click **Classifier**.
5. Review the classification and phishing score.
6. Explore the highlighted words and feature contributions.
7. Open the model performance section to inspect the evaluation metrics and confusion matrix.

The first launch may take longer because the dataset must be downloaded and the models initialized.

## Evaluating the Models

The project includes a separate evaluation script for comparing model configurations.

Run it from the project root:

```bash
python -m src.evaluate
```

The script evaluates several configurations, including:

- Logistic Regression with word-level TF-IDF.
- Multinomial Naive Bayes with word-level TF-IDF.
- Calibrated Naive Bayes.
- Logistic Regression with character-level n-grams.

It also examines the decision threshold trade-off, identifies classification errors, and generates evaluation artifacts.

### Generated outputs

After a successful evaluation, the script creates a `results/` directory containing:

| File | Description |
|---|---|
| `metrics.json` | Evaluation metrics for the compared models |
| `failures.txt` | Examples of false positives and false negatives |
| `phish_results.png` | Confusion matrix and probability calibration plots |

These files are useful for comparing models, understanding errors, and documenting experimental results.

## Evaluation Metrics

The project uses several complementary metrics to assess classification performance.

| Metric | What it measures |
|---|---|
| Precision | How many messages predicted as phishing are actually phishing |
| Recall | How many actual phishing messages are detected |
| F1-score | The harmonic mean of precision and recall |
| Accuracy | The proportion of correctly classified messages |
| Confusion matrix | The distribution of correct and incorrect predictions |
| Brier score | The quality of predicted probabilities; lower is better |
| Expected Calibration Error (ECE) | The difference between predicted confidence and observed frequencies; lower is generally better |
| ROC-AUC | How well the model ranks phishing messages above legitimate ones |

For phishing detection, recall is particularly important because a false negative means that a phishing message is incorrectly classified as legitimate.

However, maximizing recall can increase false positives. The decision threshold therefore represents a trade-off between detecting more phishing messages and avoiding unnecessary warnings.

**Note:** The 90% recall target is a validation-set policy, not a guarantee of achieving 90% recall on unseen messages or real-world traffic.

## Limitations

- Predictions depend on the dataset used to train and evaluate the models.
- Real-world messages may contain linguistic patterns that are absent from the training data.
- Legitimate messages can be flagged as phishing, while malicious messages can be missed.
- Phishing scores are model estimates, not proof that a message is malicious.
- Changing the decision threshold affects the balance between false positives and false negatives.
- The application is a decision-support tool and does not automatically block messages or replace security analysis.

**Security recommendation:** Do not rely exclusively on this classifier when deciding whether to open a link, provide credentials, or disclose sensitive information.

## Future Improvements

Potential extensions include:

- Evaluating additional machine learning models and transformer-based NLP approaches.
- Expanding the dataset with more diverse and recent phishing examples.
- Adding multilingual phishing detection.
- Improving probability calibration and threshold selection.
- Adding automated tests and experiment tracking.
- Deploying the application to a public hosting platform.
- Evaluating robustness against previously unseen phishing techniques.

## Author

**Fatima Zahra Lasfar**

- GitHub: [@FatimaZahraLasfar](https://github.com/FatimaZahraLasfar)
- Project repository: [phishing-sms-email-classifier](https://github.com/FatimaZahraLasfar/phishing-sms-email-classifier)

---

*This project demonstrates the application of NLP, classical machine learning, model evaluation, and explainability to a practical cybersecurity problem.*
