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
st.caption("Machine Learning + NLP based Fake Review Classification")

# ===============================
# HELPER FUNCTIONS
# ===============================
def clean_text(text):
    """Clean input text"""
    return re.sub(r"[^a-zA-Z ]", "", text.lower())


def extract_red_flags(text):
    """Detect suspicious patterns"""
    flags = []

    if text.count("!") >= 3:
        flags.append("Excessive exclamations")

    promo_words = [
        "buy now",
        "must buy",
        "limited offer",
        "best ever",
        "guaranteed",
        "100% working",
        "life changing"
    ]

    if any(p in text.lower() for p in promo_words):
        flags.append("Promotional language")

    if text.isupper():
        flags.append("ALL CAPS text")

    if len(text.split()) < 6:
        flags.append("Very short & unrealistic review")

    return flags


# ===============================
# USER INPUT
# ===============================
review = st.text_area("✍️ Enter product review:")

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:

        # Clean text
        clean = clean_text(review)

        # Vectorize
        vec = vectorizer.transform([clean])

        # Get probabilities
        lr_prob = lr_model.predict_proba(vec)[0]

        # Get correct class indices automatically
        classes = lr_model.classes_

        fake_index = list(classes).index(1)
        genuine_index = list(classes).index(0)

        fake_prob = lr_prob[fake_index]
        genuine_prob = lr_prob[genuine_index]

        # Extract red flags
        red_flags = extract_red_flags(review)

        # ===============================
        # CONFIDENCE DISPLAY
        # ===============================
        st.subheader("📊 Prediction Confidence")

        confidence = max(fake_prob, genuine_prob)
        st.progress(float(confidence))

        st.write(f"🔴 Fake Probability: **{fake_prob*100:.2f}%**")
        st.write(f"🟢 Genuine Probability: **{genuine_prob*100:.2f}%**")

        # ===============================
        # FINAL VERDICT LOGIC
        # ===============================
        st.subheader("✅ Final Verdict")

        if fake_prob > genuine_prob:
            verdict = "FAKE"
        else:
            verdict = "GENUINE"

        # Strong red flag override
        if verdict == "GENUINE" and len(red_flags) >= 3:
            verdict = "FAKE"

        # ===============================
        # DISPLAY RESULT
        # ===============================
        if verdict == "FAKE":

            st.error("❌ FAKE REVIEW DETECTED")

            st.markdown("**Reasons:**")

            if red_flags:
                for flag in red_flags:
                    st.write(f"- {flag}")
            else:
                st.write("- Model detected deceptive language patterns")

        else:

            st.success("✅ GENUINE REVIEW")

            st.info(
                "Reason: Higher genuine probability and realistic language patterns detected."
            )


# ===============================
# MODEL EVALUATION SECTION
# ===============================
st.divider()

st.subheader("📌 Model Evaluation")

col1, col2 = st.columns(2)

# Confusion Matrix
with col1:

    fig1, ax1 = plt.subplots(figsize=(4, 4))

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


# ROC Curve
with col2:

    fig2, ax2 = plt.subplots(figsize=(4, 4))

    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")

    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")

    ax2.legend()

    st.pyplot(fig2)
