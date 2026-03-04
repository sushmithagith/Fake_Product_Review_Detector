"""
Fake Product Review Detector - Enhanced Model Training
====================================================
Train and evaluate multiple ML models for fake review detection.
Features:
- TF-IDF vectorization with optimized parameters
- Multiple classifiers (Logistic Regression, Naive Bayes, SVM, Random Forest)
- Comprehensive evaluation metrics
- Model persistence
"""

import pandas as pd
import joblib
import os
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# CONFIGURATION
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/reviews.csv")
MODEL_DIR = BASE_DIR

# ===============================
# DATA LOADING
# ===============================
print("=" * 60)
print("FAKE REVIEW DETECTOR - MODEL TRAINING")
print("=" * 60)

print("\n📂 Loading data...")
df = pd.read_csv(DATA_PATH)

print(f"   Total reviews: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# Preprocess data
df = df[["review_body", "fake_review"]].copy()
df["review_body"] = df["review_body"].fillna("")

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text


# Enhanced cleaning: lemmatization and stopword removal (optional)
def enhanced_clean_text(text, lemmatizer=None, stop_words=None):
    text = clean_text(text)
    if not text:
        return ""
    tokens = text.split()
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

print("   Cleaning text...")

# Ensure NLTK resources are available (download if missing)
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df["cleaned_review"] = df["review_body"].apply(lambda t: enhanced_clean_text(t, lemmatizer, stop_words))

# Balance dataset (optional - uncomment to use)
# print("   Balancing dataset...")
# fake_samples = df[df['fake_review'] == 1]
# genuine_samples = df[df['fake_review'] == 0].sample(n=len(fake_samples), random_state=42)
# df = pd.concat([fake_samples, genuine_samples]).sample(frac=1, random_state=42)

X = df["cleaned_review"]
y = df["fake_review"]

print(f"   Fake reviews: {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
print(f"   Genuine reviews: {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")

# ===============================
# TRAIN-TEST SPLIT
# ===============================
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set: {len(X_test):,} samples")

# ===============================
# VECTORIZATION
# ===============================
print("\n🔧 Creating TF-IDF features...")


# Use a richer TF-IDF and dimensionality reduction (LSA) to capture semantics
vectorizer = TfidfVectorizer(
    stop_words=None,  # we removed stopwords earlier
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

svd = TruncatedSVD(n_components=300, random_state=42)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Reduce dimensionality for models that benefit from dense features
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_test_reduced = svd.transform(X_test_tfidf)

print(f"   TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"   Reduced feature matrix shape: {X_train_reduced.shape}")

# ===============================
# MODEL TRAINING
# ===============================
print("\n🤖 Training models...")

print("\n🤖 Training models with cross-validation and stacking ensemble...")

results = {}

# Prepare cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1) Tune Logistic Regression via GridSearchCV on reduced features
print("\n   Tuning Logistic Regression...")
lr_pipe = Pipeline([
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
])
lr_params = {
    'clf__C': [0.01, 0.1, 1.0, 5.0]
}
lr_search = GridSearchCV(lr_pipe, lr_params, cv=cv, scoring='f1', n_jobs=-1)
lr_search.fit(X_train_reduced, y_train)
best_lr = lr_search.best_estimator_
print(f"      Best LR params: {lr_search.best_params_}")

# 2) Tune Random Forest
print("\n   Tuning Random Forest...")
rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
rf_params = {'n_estimators': [100, 200], 'max_depth': [15, 25, None]}
rf_search = GridSearchCV(rf, rf_params, cv=cv, scoring='f1', n_jobs=-1)
rf_search.fit(X_train_reduced, y_train)
best_rf = rf_search.best_estimator_
print(f"      Best RF params: {rf_search.best_params_}")

# 3) Naive Bayes on TF-IDF (works better on sparse counts)
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_tfidf, y_train)

# 4) Gradient Boosting (as final estimator) tuned
print("\n   Tuning Gradient Boosting (final estimator)...")
gb = GradientBoostingClassifier(random_state=42)
gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
gb_search = GridSearchCV(gb, gb_params, cv=cv, scoring='f1', n_jobs=-1)
gb_search.fit(X_train_reduced, y_train)
best_gb = gb_search.best_estimator_
print(f"      Best GB params: {gb_search.best_params_}")

# 5) Build stacking ensemble (use reduced features for tree/logistic models, TF-IDF for NB)
print("\n   Building Stacking Classifier...")
estimators = [
    ('lr', best_lr.named_steps['clf'] if isinstance(best_lr, Pipeline) else best_lr),
    ('rf', best_rf),
    ('nb', nb)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=best_gb,
    cv=cv,
    n_jobs=-1,
    passthrough=False
)

# Fit stacking on reduced features for lr/rf/gb; for nb we use tfidf but stacking will receive reduced features.
stack.fit(X_train_reduced, y_train)

# Evaluate stacking
print("\n   Evaluating Stacking Classifier...")
stack_pred = stack.predict(X_test_reduced)
if hasattr(stack, 'predict_proba'):
    stack_prob = stack.predict_proba(X_test_reduced)[:, 1]
else:
    # fallback: use decision_function if available and map to [0,1]
    try:
        dec = stack.decision_function(X_test_reduced)
        stack_prob = (dec - dec.min()) / (dec.max() - dec.min())
    except Exception:
        stack_prob = np.zeros(len(stack_pred))

acc = accuracy_score(y_test, stack_pred)
prec = precision_score(y_test, stack_pred)
rec = recall_score(y_test, stack_pred)
f1 = f1_score(y_test, stack_pred)
fpr, tpr, _ = roc_curve(y_test, stack_prob)
roc_auc = auc(fpr, tpr)

results['Stacking Ensemble'] = {
    'model': stack,
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1': f1,
    'auc': roc_auc,
    'fpr': fpr,
    'tpr': tpr
}

print(f"      Accuracy: {acc:.4f}")
print(f"      Precision: {prec:.4f}")
print(f"      Recall: {rec:.4f}")
print(f"      F1: {f1:.4f}")
print(f"      AUC: {roc_auc:.4f}")

# ===============================
# SELECT BEST MODEL
# ===============================
print("\n🏆 Model Comparison:")
print("-" * 60)

best_model_name = max(results, key=lambda x: results[x]['f1'])
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
print("-" * 60)

for name, res in results.items():
    marker = "⭐" if name == best_model_name else "  "
    print(f"{marker}{name:<23} {res['accuracy']:<12.4f} {res['precision']:<12.4f} {res['recall']:<12.4f} {res['f1']:<12.4f} {res['auc']:<12.4f}")

print("-" * 60)
print(f"Best Model: {best_model_name}")

# ===============================
# SAVE MODELS
# ===============================
print("\n💾 Saving models...")

# Save vectorizer and svd
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
print("   ✓ vectorizer.pkl")
joblib.dump(svd, os.path.join(MODEL_DIR, "svd.pkl"))
print("   ✓ svd.pkl")

# Save individual models used in the stack
joblib.dump(nb, os.path.join(MODEL_DIR, "nb_model.pkl"))
print("   ✓ nb_model.pkl")
joblib.dump(best_lr, os.path.join(MODEL_DIR, "best_lr_model.pkl"))
print("   ✓ best_lr_model.pkl")
joblib.dump(best_lr, os.path.join(MODEL_DIR, "lr_model.pkl"))
print("   ✓ lr_model.pkl")
joblib.dump(best_rf, os.path.join(MODEL_DIR, "best_rf_model.pkl"))
print("   ✓ best_rf_model.pkl")
joblib.dump(best_gb, os.path.join(MODEL_DIR, "best_gb_model.pkl"))
print("   ✓ best_gb_model.pkl")

# Save stacking ensemble (primary model)
best_model = results[best_model_name]['model']
joblib.dump(best_model, os.path.join(MODEL_DIR, f"{best_model_name.replace(' ', '_').lower()}.pkl"))
print(f"   ✓ {best_model_name.replace(' ', '_').lower()}.pkl")

# Save confusion matrix (use reduced features for predictions)
best_pred = best_model.predict(X_test_reduced)
conf_matrix = confusion_matrix(y_test, best_pred)
joblib.dump(conf_matrix, os.path.join(MODEL_DIR, "confusion_matrix.pkl"))
print("   ✓ confusion_matrix.pkl")

# Save ROC data
try:
    best_proba = best_model.predict_proba(X_test_reduced)[:, 1]
except Exception:
    try:
        dec = best_model.decision_function(X_test_reduced)
        best_proba = (dec - dec.min()) / (dec.max() - dec.min())
    except Exception:
        best_proba = np.zeros(len(best_pred))

best_fpr, best_tpr, _ = roc_curve(y_test, best_proba)
best_auc = auc(best_fpr, best_tpr)
joblib.dump((best_fpr, best_tpr, best_auc), os.path.join(MODEL_DIR, "roc_data.pkl"))
print("   ✓ roc_data.pkl")

# Save classification report
classification_rep = classification_report(y_test, best_pred, target_names=['Genuine', 'Fake'])
joblib.dump(classification_rep, os.path.join(MODEL_DIR, "classification_report.pkl"))
print("   ✓ classification_report.pkl")

# Save all results
joblib.dump(results, os.path.join(MODEL_DIR, "all_results.pkl"))
print("   ✓ all_results.pkl")

# ===============================
# DETAILED EVALUATION
# ===============================
print("\n📈 Detailed Evaluation - Best Model")
print("=" * 60)

print("\nClassification Report:")
print(classification_rep)

print("\nConfusion Matrix:")
print(f"   True Negatives:  {conf_matrix[0][0]:,}")
print(f"   False Positives: {conf_matrix[0][1]:,}")
print(f"   False Negatives: {conf_matrix[1][0]:,}")
print(f"   True Positives:  {conf_matrix[1][1]:,}")

# Feature importance (for LR)
print("\n📝 Top 20 Most Important Features (Naive Bayes):")
feature_names = vectorizer.get_feature_names_out()
# MultinomialNB stores feature_log_prob_ with shape (n_classes, n_features)
try:
    log_probs = nb.feature_log_prob_
    # class 1 is 'fake' if labels are [0,1]
    fake_log_probs = log_probs[1]
    top_idx = np.argsort(fake_log_probs)[-20:][::-1]
    print("\n   Top Fake Indicators:")
    for idx in top_idx:
        print(f"      {feature_names[idx]}: {fake_log_probs[idx]:.4f}")
except Exception:
    print("   Could not extract feature importances for Naive Bayes.")

# ===============================
# COMPLETION
# ===============================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModels saved to: {MODEL_DIR}")
print("\nTo run the Streamlit app:")
print("   streamlit run app.py")
print("\nTo run the API:")
print("   cd api && uvicorn main:app --reload")
