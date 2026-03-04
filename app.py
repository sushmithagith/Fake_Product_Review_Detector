"""
Fake Product Review Detector - Enhanced Streamlit Application
=============================================================
A comprehensive ML-powered web application to detect fake product reviews.
Features:
- Real-time review analysis
- Multiple ML models
- Detailed metrics and visualizations
- Batch prediction support
- Review history tracking
"""

import streamlit as st
import joblib
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FF8E53 0%, #FF6B6B 100%);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .fake-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    .genuine-result {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    .sidebar-content {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .feature-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 10px 0;
    }
    .info-box {
        background: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #339af0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ===============================
# HELPER FUNCTIONS
# ===============================

@st.cache_data
def load_models():
    """Load all ML models and artifacts"""
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
        nb_model = joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl"))
        conf_matrix = joblib.load(os.path.join(MODEL_DIR, "confusion_matrix.pkl"))
        fpr, tpr, roc_auc = joblib.load(os.path.join(MODEL_DIR, "roc_data.pkl"))
        
        return {
            'vectorizer': vectorizer,
            'lr_model': lr_model,
            'nb_model': nb_model,
            'conf_matrix': conf_matrix,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def extract_features(text):
    """Extract various linguistic features from text"""
    features = {}
    
    # Basic features
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
    
    # Exclamation marks
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Capitalization
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['all_caps_words'] = len([w for w in text.split() if w.isupper() and len(w) > 1])
    
    # Repetition patterns
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
    features['repeated_words'] = len(re.findall(r'\b(\w+)\s+\1\b', text.lower()))
    
    # Sentiment indicators (simple)
    positive_words = ['love', 'great', 'excellent', 'amazing', 'perfect', 'best', 'awesome', 'fantastic']
    negative_words = ['hate', 'terrible', 'worst', 'awful', 'horrible', 'disappointed', 'bad', 'poor']
    
    features['positive_word_count'] = sum(1 for w in text.lower().split() if w in positive_words)
    features['negative_word_count'] = sum(1 for w in text.lower().split() if w in negative_words)
    
    # Promotional language
    promo_patterns = [
        r'buy now', r'must buy', r'limited offer', r'best ever', 
        r'guaranteed', r'100%', r'life changing', r'don\'t miss',
        r'order now', r'special price', r'deal of the day'
    ]
    features['promo_language_count'] = sum(len(re.findall(p, text.lower())) for p in promo_patterns)
    
    # Review-specific patterns
    features['has_review_title'] = 1 if len(text.split('.')) > 1 else 0
    features['is_very_short'] = 1 if len(text.split()) < 10 else 0
    features['is_very_long'] = 1 if len(text.split()) > 500 else 0
    
    return features


def detect_red_flags(text):
    """Detect suspicious patterns in reviews"""
    flags = []
    severity = []
    
    text_lower = text.lower()
    
    # Excessive exclamations
    if text.count("!") >= 3:
        flags.append("Excessive exclamation marks (3+)")
        severity.append("medium")
    
    # All caps
    if text.isupper() and len(text) > 20:
        flags.append("Entire review in CAPITAL LETTERS")
        severity.append("high")
    
    # Promotional language
    promo_words = [
        ("buy now", "Urgent buying language"),
        ("must buy", "Pressure tactics"),
        ("limited offer", "Artificial urgency"),
        ("best ever", "Exaggerated praise"),
        ("guaranteed", "False guarantees"),
        ("100% working", "Unverifiable claims"),
        ("life changing", "Overblown claims"),
        ("no risk", "Suspicious guarantee"),
        ("act now", "Pressure tactics")
    ]
    
    for pattern, description in promo_words:
        if pattern in text_lower:
            flags.append(f"Promotional: {description}")
            severity.append("medium")
    
    # Very short reviews
    if len(text.split()) < 6:
        flags.append("Unrealistically short review")
        severity.append("high")
    
    # Very long with repetition
    if len(text.split()) > 300:
        word_counts = Counter(text_lower.split())
        repeated = [w for w, c in word_counts.items() if c > 10 and len(w) > 3]
        if repeated:
            flags.append(f"Repetitive words: {', '.join(repeated[:3])}")
            severity.append("medium")
    
    # Generic review patterns
    generic_patterns = [
        (r'^(\w+\s+){0,2}stars?$', "Generic star rating only"),
        (r'^(great|good|nice|awesome|amazing|love it)\s*$', "Generic positive review"),
    ]
    
    for pattern, description in generic_patterns:
        if re.match(pattern, text_lower.strip()):
            flags.append(description)
            severity.append("low")
    
    # Suspicious characters
    if '★' in text or '☆' in text:
        flags.append("Contains star symbols (potential spam)")
        severity.append("low")
    
    return flags, severity


def predict_review(text, model_type='logistic_regression'):
    """Predict if a review is fake or genuine"""
    models = load_models()
    
    if not models['loaded']:
        return None, None, None
    
    # Clean text
    clean = clean_text(text)
    
    # Vectorize
    vec = models['vectorizer'].transform([clean])
    
    # Get predictions based on model type
    if model_type == 'logistic_regression':
        model = models['lr_model']
    else:
        model = models['nb_model']
    
    # Get probabilities
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    
    fake_index = list(classes).index(1)
    genuine_index = list(classes).index(0)
    
    fake_prob = probs[fake_index]
    genuine_prob = probs[genuine_index]
    
    prediction = 1 if fake_prob > genuine_prob else 0
    confidence = max(fake_prob, genuine_prob)
    
    return prediction, confidence, {
        'fake_prob': fake_prob,
        'genuine_prob': genuine_prob,
        'features': extract_features(text),
        'red_flags': detect_red_flags(text)
    }


# ===============================
# SIDEBAR
# ===============================
def render_sidebar():
    """Render sidebar with navigation and settings"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/null/shopping-cart.png", width=50)
        st.title("🛒 Fake Review Detector")
        
        st.markdown("---")
        
        # Navigation
        pages = {
            "🔍 Detect Reviews": "detect",
            "📊 Model Performance": "performance",
            "📝 Batch Analysis": "batch",
            "📈 Statistics": "stats",
            "ℹ️ About": "about"
        }
        
        selected_page = st.radio("Navigation", list(pages.keys()), format_func=lambda x: x)
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        model_type = st.selectbox(
            "ML Model",
            ["logistic_regression", "naive_bayes"],
            format_func=lambda x: "Logistic Regression" if x == "logistic_regression" else "Naive Bayes"
        )
        
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📉 Quick Stats")
        models = load_models()
        if models['loaded']:
            st.metric("Model AUC", f"{models['roc_auc']:.3f}")
            st.metric("Accuracy", f"{(models['conf_matrix'][0][0] + models['conf_matrix'][1][1]) / models['conf_matrix'].sum():.1%}")
        
        return pages[selected_page], model_type, confidence_threshold


# ===============================
# MAIN PAGE - DETECT REVIEWS
# ===============================
def render_detect_page(model_type, confidence_threshold):
    """Render the main detection page"""
    
    # Header
    st.title("🔍 Fake Review Detection")
    st.markdown("Enter a product review below to analyze if it's genuine or fake")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        review_text = st.text_area(
            "✍️ Enter Product Review:",
            height=150,
            placeholder="Paste or type the review you want to analyze..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze Review", use_container_width=True)
        
        # Quick examples
        st.markdown("### Quick Examples")
        example_reviews = {
            "Promotional Fake": "BEST PRODUCT EVER! You MUST buy now! Limited offer, guaranteed 100% working! Life changing! Buy now!",
            "Generic Fake": "Five Stars. Great! Five Stars. Love it!",
            "Genuine Review": "I purchased this last month and have been using it daily. The quality is good but there are some issues with durability after repeated use."
        }
        example_choice = st.selectbox("Try an example:", ["Custom", "Promotional Fake", "Generic Fake", "Genuine Review"])
        if example_choice != "Custom":
            review_text = example_reviews[example_choice]
    
    # Analysis results
    if analyze_btn and review_text.strip():
        with st.spinner("Analyzing review..."):
            prediction, confidence, details = predict_review(review_text, model_type)
            
            if prediction is None:
                st.error("⚠️ Models not loaded. Please ensure model files exist in the 'model/' directory.")
                return
            
            fake_prob = details['fake_prob']
            genuine_prob = details['genuine_prob']
            red_flags, severity = details['red_flags']
            
            # Override with threshold
            if fake_prob > confidence_threshold:
                prediction = 1
            elif genuine_prob > confidence_threshold:
                prediction = 0
            
            # Results display
            st.markdown("---")
            
            # Main prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-box fake-result">
                        ❌ FAKE REVIEW<br>DETECTED
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-box genuine-result">
                        ✅ GENUINE<br>REVIEW
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Confidence Score")
                st.progress(float(confidence))
                st.markdown(f"**Confidence: {confidence*100:.1f}%**")
                
                if prediction == 1:
                    st.markdown(f"🤖 AI is **{confidence*100:.1f}%** sure this is a **fake** review")
                else:
                    st.markdown(f"🤖 AI is **{confidence*100:.1f}%** sure this is a **genuine** review")
            
            with col3:
                st.markdown("### 📈 Probability Breakdown")
                
                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=[genuine_prob * 100, fake_prob * 100],
                    y=['Genuine', 'Fake'],
                    orientation='h',
                    marker_color=['#51cf66', '#ff6b6b'],
                    text=[f'{genuine_prob*100:.1f}%', f'{fake_prob*100:.1f}%'],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    showlegend=False,
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Detailed analysis
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🚩 Red Flags Detected")
                if red_flags:
                    for i, (flag, sev) in enumerate(zip(red_flags, severity)):
                        if sev == "high":
                            color = "🔴"
                        elif sev == "medium":
                            color = "🟠"
                        else:
                            color = "🟡"
                        st.markdown(f"- {color} **{flag}**")
                else:
                    st.success("✓ No suspicious patterns detected")
                
                st.markdown("### 📝 Text Features")
                features = details['features']
                st.write(f"- Word Count: {features['word_count']}")
                st.write(f"- Character Count: {features['char_count']}")
                st.write(f"- Exclamation Marks: {features['exclamation_count']}")
                st.write(f"- All Caps Words: {features['all_caps_words']}")
                st.write(f"- Promotional Language: {features['promo_language_count']}")
            
            with col2:
                st.markdown("### 💡 Why This Prediction?")
                
                reasons = []
                
                if prediction == 1:  # Fake
                    if features['exclamation_count'] >= 3:
                        reasons.append("High number of exclamation marks suggests emotional manipulation")
                    if features['all_caps_words'] >= 3:
                        reasons.append("Multiple ALL CAPS words indicate unnatural writing")
                    if features['promo_language_count'] >= 2:
                        reasons.append("Contains promotional/pressure language")
                    if features['is_very_short']:
                        reasons.append("Review is unrealistically short")
                    if len(red_flags) >= 3:
                        reasons.append(f"Multiple red flags detected ({len(red_flags)} total)")
                    
                    if not reasons:
                        reasons.append("ML model detected patterns typical of fake reviews")
                else:  # Genuine
                    if features['word_count'] > 20:
                        reasons.append("Detailed review with substantial content")
                    if features['is_very_short'] == 0:
                        reasons.append("Normal-length detailed review")
                    if len(red_flags) == 0:
                        reasons.append("No suspicious patterns detected")
                    
                    if not reasons:
                        reasons.append("ML model detected natural writing patterns")
                
                for reason in reasons:
                    st.write(f"• {reason}")
            
            # Original text
            with st.expander("📄 View Original Review"):
                st.write(review_text)
    
    elif analyze_btn and not review_text.strip():
        st.warning("⚠️ Please enter a review to analyze")


# ===============================
# PERFORMANCE PAGE
# ===============================
def render_performance_page():
    """Render model performance metrics"""
    st.title("📊 Model Performance")
    
    models = load_models()
    
    if not models['loaded']:
        st.error("⚠️ Models not loaded")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    conf_matrix = models['conf_matrix']
    tn, fp, fn, tp = conf_matrix.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    with col4:
        st.metric("F1 Score", f"{f1:.2%}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔢 Confusion Matrix")
        
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Genuine (0)", "Fake (1)"],
            yticklabels=["Genuine (0)", "Fake (1)"],
            ax=ax1,
            annot_kws={"size": 16}
        )
        ax1.set_xlabel("Predicted Label", fontsize=12)
        ax1.set_ylabel("True Label", fontsize=12)
        ax1.set_title("Confusion Matrix", fontsize=14)
        
        st.pyplot(fig1)
    
    with col2:
        st.markdown("### 📈 ROC Curve")
        
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        
        ax2.plot(models['fpr'], models['tpr'], 
                 label=f'ROC Curve (AUC = {models["roc_auc"]:.3f})',
                 color='#FF6B6B', linewidth=2)
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        ax2.fill_between(models['fpr'], models['tpr'], alpha=0.3, color='#FF6B6B')
        
        ax2.set_xlabel("False Positive Rate", fontsize=12)
        ax2.set_ylabel("True Positive Rate", fontsize=12)
        ax2.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
    
    # Detailed metrics table
    st.markdown("---")
    st.markdown("### 📋 Detailed Metrics")
    
    metrics_df = pd.DataFrame({
        'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives',
                   'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Value': [tn, fp, fn, tp, accuracy, precision, recall, f1, models['roc_auc']],
        'Description': [
            'Correctly identified genuine reviews',
            'Genuine reviews incorrectly marked as fake',
            'Fake reviews incorrectly marked as genuine',
            'Correctly identified fake reviews',
            'Overall prediction accuracy',
            'Ratio of true positives to predicted positives',
            'Ratio of true positives to actual positives',
            'Harmonic mean of precision and recall',
            'Area under ROC curve'
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True)


# ===============================
# BATCH ANALYSIS PAGE
# ===============================
def render_batch_page():
    """Render batch analysis page"""
    st.title("📝 Batch Review Analysis")
    
    st.markdown("""
    <div class="info-box">
    Upload a CSV file with reviews to analyze multiple reviews at once.
    The CSV should have a column named 'review_body' or 'review' containing the review text.
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find review column
            review_col = None
            for col in ['review_body', 'review', 'text', 'review_text']:
                if col in df.columns:
                    review_col = col
                    break
            
            if review_col is None:
                st.error("Could not find review column. Please ensure your CSV has a column named 'review_body', 'review', 'text', or 'review_text'")
                return
            
            st.success(f"✓ Found {len(df)} reviews in the file")
            st.write("Preview:")
            st.dataframe(df.head())
            
            # Analyze button
            if st.button("🔍 Analyze All Reviews"):
                with st.spinner("Analyzing reviews..."):
                    models = load_models()
                    
                    if not models['loaded']:
                        st.error("⚠️ Models not loaded")
                        return
                    
                    results = []
                    
                    for idx, row in df.iterrows():
                        text = str(row[review_col])
                        prediction, confidence, details = predict_review(text)
                        
                        if prediction is not None:
                            results.append({
                                'review': text[:100] + '...' if len(text) > 100 else text,
                                'prediction': 'Fake' if prediction == 1 else 'Genuine',
                                'confidence': f"{confidence*100:.1f}%",
                                'fake_prob': details['fake_prob'],
                                'genuine_prob': details['genuine_prob'],
                                'red_flags_count': len(details['red_flags'][0])
                            })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    fake_count = (results_df['prediction'] == 'Fake').sum()
                    genuine_count = (results_df['prediction'] == 'Genuine').sum()
                    avg_confidence = results_df['confidence'].str.rstrip('%').astype(float).mean()
                    
                    with col1:
                        st.metric("Total Reviews", len(results_df))
                    with col2:
                        st.metric("Fake Reviews", fake_count, delta_color="inverse")
                    with col3:
                        st.metric("Genuine Reviews", genuine_count)
                    with col4:
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Results table
                    st.markdown("### 📋 Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "fake_review_analysis.csv",
                        "text/csv"
                    )
                    
                    # Visualization
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            values=[fake_count, genuine_count],
                            names=['Fake', 'Genuine'],
                            title='Distribution of Reviews',
                            color_discrete_sequence=['#ff6b6b', '#51cf66']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig2 = px.histogram(
                            results_df,
                            x='confidence',
                            nbins=20,
                            title='Confidence Distribution',
                            color_discrete_sequence=['#FF6B6B']
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ===============================
# STATISTICS PAGE
# ===============================
def render_stats_page():
    """Render statistics page"""
    st.title("📈 Statistics & Insights")
    
    # Load data
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "data/reviews.csv"))
        
        # Basic stats
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_reviews = len(df)
        fake_reviews = (df['fake_review'] == 1).sum()
        genuine_reviews = (df['fake_review'] == 0).sum()
        
        with col1:
            st.metric("Total Reviews", f"{total_reviews:,}")
        with col2:
            st.metric("Fake Reviews", f"{fake_reviews:,}", delta_color="inverse")
        with col3:
            st.metric("Genuine Reviews", f"{genuine_reviews:,}")
        with col4:
            st.metric("Fake Percentage", f"{fake_reviews/total_reviews*100:.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Review Distribution")
            fig = px.pie(
                values=[genuine_reviews, fake_reviews],
                names=['Genuine', 'Fake'],
                title='Genuine vs Fake Reviews',
                color_discrete_sequence=['#51cf66', '#ff6b6b'],
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Review Length Distribution")
            
            if 'review_body' in df.columns:
                df['review_length'] = df['review_body'].fillna('').str.split().str.len()
                
                fig2 = px.histogram(
                    df,
                    x='review_length',
                    color='fake_review',
                    nbins=50,
                    title='Review Length by Type',
                    color_discrete_sequence=['#51cf66', '#ff6b6b'],
                    labels={0: 'Genuine', 1: 'Fake'}
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Word cloud for fake vs genuine
        st.markdown("---")
        st.markdown("### ☁️ Word Cloud Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Genuine Reviews - Common Words")
            genuine_text = ' '.join(df[df['fake_review'] == 0]['review_body'].fillna('').tolist())
            
            # Simple word frequency
            words = genuine_text.lower().split()
            word_counts = Counter(words)
            common_genuine = word_counts.most_common(30)
            
            if common_genuine:
                words_df = pd.DataFrame(common_genuine, columns=['Word', 'Count'])
                fig = px.bar(words_df, x='Count', y='Word', orientation='h',
                            title='Top 30 Words in Genuine Reviews',
                            color_discrete_sequence=['#51cf66'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Fake Reviews - Common Words")
            fake_text = ' '.join(df[df['fake_review'] == 1]['review_body'].fillna('').tolist())
            
            words = fake_text.lower().split()
            word_counts = Counter(words)
            common_fake = word_counts.most_common(30)
            
            if common_fake:
                words_df = pd.DataFrame(common_fake, columns=['Word', 'Count'])
                fig = px.bar(words_df, x='Count', y='Word', orientation='h',
                            title='Top 30 Words in Fake Reviews',
                            color_discrete_sequence=['#ff6b6b'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Sample reviews
        st.markdown("---")
        st.markdown("### 📝 Sample Reviews")
        
        tab1, tab2 = st.tabs(["Genuine Reviews", "Fake Reviews"])
        
        with tab1:
            genuine_samples = df[df['fake_review'] == 0]['review_body'].dropna().head(5)
            for i, review in enumerate(genuine_samples, 1):
                with st.expander(f"Sample {i}"):
                    st.write(review[:500] + "..." if len(review) > 500 else review)
        
        with tab2:
            fake_samples = df[df['fake_review'] == 1]['review_body'].dropna().head(5)
            for i, review in enumerate(fake_samples, 1):
                with st.expander(f"Sample {i}"):
                    st.write(review[:500] + "..." if len(review) > 500 else review)
                    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


# ===============================
# ABOUT PAGE
# ===============================
def render_about_page():
    """Render about page"""
    st.title("ℹ️ About")
    
    st.markdown("""
    ## 🛒 Fake Product Review Detector
    
    An advanced machine learning application designed to identify fake product reviews
    on e-commerce platforms.
    
    ### 🎯 Features
    
    - **Real-time Analysis**: Instantly detect fake reviews with high accuracy
    - **Multiple Models**: Choose between Logistic Regression and Naive Bayes
    - **Detailed Metrics**: View comprehensive performance metrics and visualizations
    - **Batch Processing**: Analyze multiple reviews at once
    - **Pattern Detection**: Identify common fake review patterns and red flags
    
    ### 🔬 How It Works
    
    1. **Text Preprocessing**: Clean and normalize the review text
    2. **Feature Extraction**: Extract TF-IDF features from the text
    3. **ML Prediction**: Use trained classification models to predict authenticity
    4. **Red Flag Analysis**: Check for suspicious patterns and keywords
    
    ### 📊 Models Used
    
    - **Logistic Regression**: Primary classifier with high accuracy
    - **Naive Bayes**: Alternative classifier for ensemble predictions
    
    ### 🚀 Deployment
    
    This application can be deployed on Streamlit Cloud, Heroku, or any
    Python-compatible hosting platform.
    
    ### 📁 Project Structure
    
    
```
    Fake_Product_Review_Detector/
    ├── app.py                  # Main Streamlit application
    ├── requirements.txt        # Python dependencies
    ├── model/                  # Trained models
    │   ├── vectorizer.pkl
    │   ├── lr_model.pkl
    │   ├── nb_model.pkl
    │   ├── confusion_matrix.pkl
    │   └── roc_data.pkl
    └── data/                   # Training data
        └── reviews.csv
    
```
    
    ---
    
    **Version**: 2.0.0  
    **Last Updated**: 2024
    """)
    
    # Credits
    st.markdown("### 🙏 Credits")
    st.markdown("""
    - Built with Streamlit
    - Machine Learning powered by Scikit-learn
    - Visualizations using Plotly and Matplotlib
    """)


# ===============================
# MAIN
# ===============================
def main():
    """Main application entry point"""
    
    # Load models
    models = load_models()
    
    if not models['loaded']:
        st.error(f"""
        ⚠️ **Models Not Loaded**
        
        Please ensure the model files exist in the 'model/' directory.
        
        Error: {models.get('error', 'Unknown error')}
        
        Run `python model/train_model.py` to train and save the models.
        """)
        return
    
    # Render sidebar and get current page
    current_page, model_type, confidence_threshold = render_sidebar()
    
    # Render current page
    if current_page == "detect":
        render_detect_page(model_type, confidence_threshold)
    elif current_page == "performance":
        render_performance_page()
    elif current_page == "batch":
        render_batch_page()
    elif current_page == "stats":
        render_stats_page()
    elif current_page == "about":
        render_about_page()


if __name__ == "__main__":
    main()
