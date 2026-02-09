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

        # IMPORTANT: correct probability mapping
        genuine_prob = lr_prob[0]
        fake_prob = lr_prob[1]

        red_flags = extract_red_flags(review)

        # ===============================
        # CONFIDENCE BARS
        # ===============================
        st.subheader("📊 Prediction Confidence")

        if fake_prob > genuine_prob:
            st.progress(float(fake_prob))
        else:
            st.progress(float(genuine_prob))

        st.write(f"Fake: **{fake_prob*100:.2f}%**")
        st.write(f"Genuine: **{genuine_prob*100:.2f}%**")

        # ===============================
        # FINAL VERDICT (FIXED)
        # ===============================
        st.subheader("✅ Final Verdict")

        # Base decision on probability
        if fake_prob > genuine_prob:
            verdict = "FAKE"
        else:
            verdict = "GENUINE"

        # Optional override only if VERY strong signals
        if verdict == "GENUINE" and len(red_flags) >= 3:
            verdict = "FAKE"

        # Display result
        if verdict == "FAKE":
            st.error("❌ FAKE REVIEW")

            st.markdown("**Reason:**")
            if red_flags:
                for r in red_flags:
                    st.write(f"- {r}")
            else:
                st.write("- Model detected deceptive patterns")

        else:
            st.success("✅ GENUINE REVIEW")
            st.info("Reason: Higher genuine probability and realistic language")
