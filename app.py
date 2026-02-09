import streamlit as st
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))

conf_matrix = joblib.load(os.path.join(MODEL_DIR, "confusion_matrix.pkl"))
fpr, tpr, roc_auc = joblib.load(os.path.join(MODEL_DIR, "roc_data.pkl"))

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🛒 Fake Product Review Detector")
st.caption("Machine Learning + NLP")

# ===============================
# HELPER FUNCTIONS
# ===============================
def clean_text(text):
    return re.sub(r"[^a-zA-Z ]", "", text.lower())

def extract_red_flags(text):
    flags = []

    if text.count("!") >= 3:
        flags.append("Excessive exclamations")

    promo_words = ["buy now", "must buy", "limited offer", "best ever", "guaranteed"]
    if any(p in text.lower() for p in promo_words):
        flags.append("Promotional language")

    if text.isupper():
        flags.append("ALL CAPS text")

    if len(text.split()) < 6:
        flags.append("Very short & unrealistic review")

    return flags

# ===============================
# INPUT
# ===============================
review = st.text_area("✍️ Enter product review")

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        clean = clean_text(review)
        vec = vectorizer.transform([clean])

        lr_prob = lr_model.predict_proba(vec)[0]
        lr_pred = lr_model.predict(vec)[0]

        red_flags = extract_red_flags(review)

        # ===============================
        # CONFIDENCE BARS
        # ===============================
        st.subheader("📊 Prediction Confidence")
        st.progress(float(lr_prob[1]))
        st.write(f"Fake: **{lr_prob[1]*100:.2f}%**")
        st.write(f"Genuine: **{lr_prob[0]*100:.2f}%**")

        # ===============================
        # FINAL VERDICT LOGIC
        # ===============================
        st.subheader("✅ Final Verdict")

        # Override ML if strong fake signals
        if lr_pred == 1 or len(red_flags) >= 2:
            st.error("❌ FAKE REVIEW")

            st.markdown("**Reason:**")
            for r in red_flags:
                st.write(f"- {r}")

            if not red_flags:
                st.write("- Model detected deceptive language patterns")

        else:
            st.success("✅ GENUINE REVIEW")
            st.info("Reason: Balanced opinion and realistic user experience")

# ===============================
# VISUAL EVALUATION
# ===============================
st.divider()
st.subheader("📌 Model Evaluation")

col1, col2 = st.columns(2)

# Confusion Matrix (smaller)
with col1:
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Genuine", "Fake"],
        yticklabels=["Genuine", "Fake"],
        ax=ax1
    )
    ax1.set_title("Confusion Matrix")
    st.pyplot(fig1)

# ROC Curve (smaller)
with col2:
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)
