import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/reviews.csv")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
df = df[["review_body", "fake_review"]]
df["review_body"] = df["review_body"].fillna("")

X = df["review_body"]
y = df["fake_review"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# VECTORIZE
# ===============================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# MODELS
# ===============================
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

nb_model.fit(X_train_vec, y_train)
lr_model.fit(X_train_vec, y_train)

# ===============================
# METRICS
# ===============================
lr_preds = lr_model.predict(X_test_vec)
cm = confusion_matrix(y_test, lr_preds)

lr_probs = lr_model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, _ = roc_curve(y_test, lr_probs)
roc_auc = auc(fpr, tpr)

# ===============================
# SAVE EVERYTHING
# ===============================
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))
joblib.dump(nb_model, os.path.join(BASE_DIR, "nb_model.pkl"))
joblib.dump(lr_model, os.path.join(BASE_DIR, "lr_model.pkl"))

joblib.dump(cm, os.path.join(BASE_DIR, "confusion_matrix.pkl"))
joblib.dump((fpr, tpr, roc_auc), os.path.join(BASE_DIR, "roc_data.pkl"))

print("✅ Training completed & metrics saved")

