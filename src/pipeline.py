"""Lab 04 - Phish or Legit?: TF-IDF + Logistic Regression (no UI code here)."""
import re

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SEED = 42
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b\w+\.(?:com|org|info|ly)\S*", re.I)
MONEY_RE = re.compile(r"\$\s?\d+(?:[.,]\d+)?")
NUM_RE = re.compile(r"\d+")


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data():
    """Charge le dataset équilibré MENTHOS-Spam depuis Hugging Face (CSV)."""
    repo_id = "LHRS-UM-FERI/MENTHOS-dataset-spam"
    files = {
        "train": "spam_train.csv",
        "validation": "spam_val.csv",
        "test": "spam_test.csv",
    }
    dfs = []
    for split_name, filename in files.items():
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset"
        )
        dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)

    if "text" not in df.columns:
        raise ValueError("Le dataset doit contenir une colonne 'text'.")
    if "label" not in df.columns:
        raise ValueError("Le dataset doit contenir une colonne 'label'.")

    df["y"] = (df["label"].astype(int) > 0).astype(int)
    return df


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def split(df, test_size=0.25):
    """Train / test split (kept for backward compatibility)."""
    return train_test_split(
        df["text"], df["y"], test_size=test_size, stratify=df["y"], random_state=SEED
    )


def split_train_val_test(df, test_size=0.20, val_size=0.20):
    """
    Train / validation / test split, stratified.
    - train : fit the model
    - val   : choose the threshold (policy decision)
    - test  : evaluate once (unbiased report)
    """
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        df["text"], df["y"], test_size=test_size, stratify=df["y"], random_state=SEED
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=SEED
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ----------------------------------------------------------------------
# Threshold selection
# ----------------------------------------------------------------------
def select_threshold(y_val, p_val, min_recall=0.90):
    """
    Choose a decision threshold on the VALIDATION set according to a policy:
    "catch at least min_recall of phishing messages".

    Returns (threshold, precision_at_threshold, recall_at_threshold).
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    precision, recall = precision[:-1], recall[:-1]  # drop the last (no-threshold) point

    # Points where we meet the policy
    valid = recall >= min_recall
    if not valid.any():
        # Policy unreachable: fall back to best F1
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        i = int(np.argmax(f1))
    else:
        # Among valid points, pick the one with highest precision
        valid_idx = np.where(valid)[0]
        best = valid_idx[np.argmax(precision[valid_idx])]
        i = int(best)

    th = float(thresholds[i])
    return th, float(precision[i]), float(recall[i])


# ----------------------------------------------------------------------
# Preprocessing / model
# ----------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" urltok ", text)
    text = MONEY_RE.sub(" moneytok ", text)
    return NUM_RE.sub(" numtok ", text)


def make_pipeline(kind="lr", analyzer="word"):
    if analyzer == "word":
        tfidf = TfidfVectorizer(
            preprocessor=preprocess, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
    else:
        tfidf = TfidfVectorizer(
            preprocessor=preprocess, analyzer="char_wb",
            ngram_range=(2, 5), min_df=3, sublinear_tf=True,
        )

    if kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    elif kind == "lr":
        clf = LogisticRegression(
            max_iter=1000, C=3.0, random_state=SEED, class_weight="balanced"
        )
    elif kind == "nb_cal":
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(MultinomialNB(alpha=0.1), method="sigmoid", cv=5)
    elif kind == "lr_cal":
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(
            LogisticRegression(
                max_iter=1000, C=3.0, random_state=SEED, class_weight="balanced"
            ),
            method="sigmoid", cv=5,
        )
    else:
        raise ValueError(kind)

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ----------------------------------------------------------------------
# Prediction & explanation
# ----------------------------------------------------------------------
def predict(pipe, text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The message is empty - paste some text to classify.")
    score = float(pipe.predict_proba([text])[0][1])
    return ("phish" if score >= 0.5 else "legit"), score


def contributions(pipe, text: str):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    X = vec.transform([text])
    names = vec.get_feature_names_out()
    row = X.tocoo()
    contrib = {names[j]: float(v * clf.coef_[0][j]) for j, v in zip(row.col, row.data)}
    return dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1])))


def word_weights(pipe, text: str):
    contrib = contributions(pipe, text)
    out = {}
    for w in re.findall(r"\w+", preprocess(text)):
        out[w] = contrib.get(w, 0.0)
    return out


def global_top_features(pipe, k=8):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names, coef = vec.get_feature_names_out(), clf.coef_[0]
    order = np.argsort(coef)
    pos = [(names[i], round(float(coef[i]), 2)) for i in order[::-1][:k]]
    neg = [(names[i], round(float(coef[i]), 2)) for i in order[:k]]
    return pos, neg


def get_misclassified_examples(pipe, Xte, yte, n_per_type=2, threshold=0.5):
    p = pipe.predict_proba(Xte)[:, 1]
    pred = (p >= threshold).astype(int)
    fn_idx = np.where((yte == 1) & (pred == 0))[0]
    fp_idx = np.where((yte == 0) & (pred == 1))[0]

    examples = {}
    for i, idx in enumerate(fn_idx[:n_per_type]):
        examples[f"Faux négatif {i + 1} (phishing raté)"] = Xte.iloc[idx]
    for i, idx in enumerate(fp_idx[:n_per_type]):
        examples[f"Faux positif {i + 1} (légitime flaggé)"] = Xte.iloc[idx]
    return examples


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)