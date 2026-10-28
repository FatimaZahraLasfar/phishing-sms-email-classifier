"""Lab 04 - Phish or Legit? (Streamlit UI). Run: streamlit run app.py"""
import re

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.pipeline import (
    load_data,
    split_train_val_test,          # <-- NEW
    select_threshold,              # <-- NEW
    make_pipeline,
    predict,
    word_weights,
    contributions,
    get_misclassified_examples,
    ece,
)

st.set_page_config(page_title="Phish or Legit?", layout="wide")
st.title("Phish or Legit?")
st.caption(
    "Classification de messages (phishing vs légitime) sur un dataset réel et équilibré "
    "(Hugging Face: LHRS-UM-FERI/MENTHOS-dataset-spam)."
)

st.info(
    "Cet outil aide à la décision humaine et ne bloque pas les messages automatiquement. "
    "Le jeu de données utilisé est réel (source : Hugging Face) et équilibré, "
    "mais les performances sur du trafic réel peuvent être inférieures."
)


# ======================================================================
# Chargement : train/val/test + sélection du seuil sur la validation
# ======================================================================
@st.cache_resource
def load_models_and_artifacts():
    df = load_data()
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_train_val_test(df)  # <-- NEW
    y_va = np.asarray(y_va)
    y_te = np.asarray(y_te)

    # Fit on TRAIN only
    model = make_pipeline("lr").fit(X_tr, y_tr)
    model_cal = make_pipeline("lr_cal").fit(X_tr, y_tr)

    # ---- Threshold selection on VALIDATION (policy: recall >= 0.90) ----
    p_val = model.predict_proba(X_va)[:, 1]                       # <-- NEW
    policy_threshold, val_prec, val_rec = select_threshold(       # <-- NEW
        y_va, p_val, min_recall=0.90                              # <-- NEW
    )                                                              # <-- NEW

    # ---- Metrics computed on TEST at the policy threshold ----
    p_test = model.predict_proba(X_te)[:, 1]
    pred_test = (p_test >= policy_threshold).astype(int)          # <-- NEW (was 0.5)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_te, pred_test, average="binary", zero_division=0
    )
    acc = float((pred_test == y_te).mean())
    brier = brier_score_loss(y_te, p_test)
    ece_val = ece(y_te, p_test)
    roc_auc = roc_auc_score(y_te, p_test)

    metrics = {
        "Precision (phish)": pr,
        "Recall (phish)": rc,
        "F1 (phish)": f1,
        "Accuracy": acc,
        "Brier score": brier,
        "ECE": ece_val,
        "ROC-AUC": roc_auc,
    }

    cm = confusion_matrix(y_te, pred_test)

    misclassified = get_misclassified_examples(
        model, X_te, y_te, n_per_type=2, threshold=policy_threshold  # <-- NEW
    )

    return (
        model, model_cal, X_te, y_te,
        misclassified_examples if False else misclassified,
        metrics, cm, p_test,
        policy_threshold, val_prec, val_rec,                     # <-- NEW
    )


(
    model, model_cal, Xte, yte,
    misclassified_examples,
    metrics, cm, p_test,
    POLICY_THRESHOLD, VAL_PREC, VAL_REC,                          # <-- NEW
) = load_models_and_artifacts()

# ======================================================================
# Section 3 : Classification interactive
# ======================================================================
st.divider()
st.subheader(" Classifier un message")

EXAMPLES = {
    "Phishing typique (exemple)": (
        "Dear customer, your account has been locked. "
        "Verify now http://bit.ly/ab12cd"
    ),
    "Légitime typique (exemple)": (
        "Hi, your appointment is tomorrow at 9. "
        "Details are in your online account."
    ),
    **misclassified_examples,
}

if "message_text" not in st.session_state:
    st.session_state.message_text = list(EXAMPLES.values())[0]


def load_example():
    st.session_state.message_text = EXAMPLES[st.session_state.example_choice]


st.selectbox(
    "Charger un message d'exemple",
    list(EXAMPLES.keys()),
    key="example_choice",
    on_change=load_example,
)

text = st.text_area("Collez un message :", key="message_text", height=120)

# --- Slider : default = policy threshold --------------------------------
threshold = st.slider(                                            # <-- NEW default
    f"Seuil de décision (le seuil de politique = {POLICY_THRESHOLD:.2f})",
    min_value=0.05, max_value=0.95,
    value=float(POLICY_THRESHOLD), step=0.05,
)

if st.button("Classifier", type="primary"):
    st.session_state.classify = True

if st.session_state.get("classify"):
    try:
        _, score = predict(model, text)
        _, cal_score = predict(model_cal, text)
        label = "phish" if score >= threshold else "legit"
    except ValueError as err:
        st.error(str(err))
    else:
        if label == "phish":
            st.error("###  Verdict : PHISHING")
        else:
            st.success("###  Verdict : LÉGITIME")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Score phishing (brut)",
                value=f"{score * 100:.1f}%",
                delta=f"seuil {threshold * 100:.0f}%",
                delta_color="off",
            )
        with c2:
            st.metric(label="Score calibré", value=f"{cal_score * 100:.1f}%")
        with c3:
            st.metric(label="Décision", value=label.upper())

        st.markdown("**Score phishing par rapport au seuil**")
        st.progress(
            min(max(score, 0.0), 1.0),
            text=f"{score * 100:.1f}%  (seuil = {threshold * 100:.0f}%)",
        )

        st.markdown("**Texte annoté (rouge = pousse vers le phishing, vert = vers le légitime)**")
        weights = word_weights(model, text)
        parts = []
        for tok in re.findall(r"\w+|\W+", text):
            w = weights.get(tok.lower(), 0.0)
            if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
                parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
            else:
                parts.append(tok)
        st.markdown("".join(parts))

        contrib = contributions(model, text)
        positive = [(k, v) for k, v in contrib.items() if v > 0][:6]
        negative = [(k, v) for k, v in contrib.items() if v < 0][:6]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Caractéristiques poussant vers le phishing**")
            if positive:
                st.dataframe(
                    {"feature": [t for t, _ in positive],
                     "contribution (log-odds)": [round(v, 2) for _, v in positive]},
                    hide_index=True, use_container_width=True,
                )
            else:
                st.caption("Aucune caractéristique positive forte.")
        with col2:
            st.markdown("**Caractéristiques poussant vers le légitime**")
            if negative:
                st.dataframe(
                    {"feature": [t for t, _ in negative],
                     "contribution (log-odds)": [round(v, 2) for _, v in negative]},
                    hide_index=True, use_container_width=True,
                )
            else:
                st.caption("Aucune caractéristique négative forte.")


# ======================================================================
# Section 1 : Performances du modèle
# ======================================================================
with st.expander(" Performances du modèle (métriques + matrice de confusion)", expanded=False):
    st.markdown(
        f"**Seuil de politique (validation) :** `{POLICY_THRESHOLD:.2f}`  "
        f"· précision val = {VAL_PREC:.2f} · rappel val = {VAL_REC:.2f}"
    )
    st.caption(
        "Politique : rappel ≥ 90 % sur validation. "
        "Évalué une seule fois sur le jeu de test."
    )

    col_m, col_cm = st.columns([1.3, 1])

    # ---- Métriques compactes ----
    with col_m:
        st.markdown("**Métriques (jeu de test)**")
        left, right = st.columns(2)
        items = list(metrics.items())
        half = (len(items) + 1) // 2
        for k, v in items[:half]:
            left.markdown(f"**{k}**  \n`{v:.3f}`")
        for k, v in items[half:]:
            right.markdown(f"**{k}**  \n`{v:.3f}`")

    # ---- Matrice de confusion compacte ----
    with col_cm:
        st.markdown("**Matrice de confusion**")
        fig, ax = plt.subplots(figsize=(2.8, 2.2))
        ax.imshow(cm, cmap="Greys")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, int(cm[i, j]),
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["prédit légitime", "prédit phish"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["vrai légitime", "vrai phish"], fontsize=8)
        ax.tick_params(length=0)
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=False)

        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN={tn} · FP={fp} · FN={fn} · TP={tp}")

# ======================================================================
# Section 3 : Précision / Rappel vs seuil
# ======================================================================
st.divider()
st.subheader(" Précision / Rappel en fonction du seuil")

precision, recall, thresholds = precision_recall_curve(yte, p_test)
fig, ax = plt.subplots(figsize=(4.5, 2.8))
ax.plot(thresholds, precision[:-1], label="Précision", color="blue", lw=1.2)
ax.plot(thresholds, recall[:-1], label="Rappel", color="orange", lw=1.2)
ax.axvline(POLICY_THRESHOLD, color="green", linestyle="--", lw=1,
           label=f"Seuil politique ({POLICY_THRESHOLD:.2f})")
ax.axvline(threshold, color="red", linestyle=":", lw=1,
           label=f"Seuil slider ({threshold:.2f})")
ax.set_xlabel("Seuil de décision", fontsize=8)
ax.set_ylabel("Score", fontsize=8)
ax.set_title("Précision / Rappel vs seuil (test)", fontsize=9)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc="best")
ax.grid(True, alpha=0.3, lw=0.5)
fig.tight_layout(pad=0.3)
st.pyplot(fig, use_container_width=False)   