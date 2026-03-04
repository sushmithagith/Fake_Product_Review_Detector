# 🛒 Fake Product Review Detector

A comprehensive Machine Learning project to detect fake product reviews on e-commerce platforms. Built with Streamlit for the frontend and FastAPI for the backend.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)

## 🚀 Features

- **Real-time Review Analysis**: Instantly detect fake reviews with ML-powered predictions
- **Multiple ML Models**: Choose between Logistic Regression, Naive Bayes, SVM, and Random Forest
- **Detailed Metrics**: Comprehensive performance visualization (Confusion Matrix, ROC Curve)
- **Batch Processing**: Analyze multiple reviews at once via CSV upload
- **Pattern Detection**: Identify red flags and suspicious patterns in reviews
- **REST API**: Production-ready FastAPI backend for integration
- **Statistics Dashboard**: Explore dataset insights and word analysis

## 📁 Project Structure

```
Fake_Product_Review_Detector/
├── app.py                      # Main Streamlit application
├── README.md                   # This file
├── requirements.txt           # Streamlit app dependencies
├── api/
│   ├── main.py               # FastAPI backend
│   └── requirements.txt      # API dependencies
├── model/
│   ├── train_model.py       # Model training script
│   ├── vectorizer.pkl       # TF-IDF vectorizer
│   ├── lr_model.pkl         # Logistic Regression model
│   ├── nb_model.pkl         # Naive Bayes model
│   ├── confusion_matrix.pkl
│   └── roc_data.pkl
└── data/
    └── reviews.csv          # Training dataset (~130K reviews)
```

## 🛠️ Installation

### 1. Clone the Repository

```
bash
git clone <repository-url>
cd Fake_Product_Review_Detector
```

### 2. Create Virtual Environment

```
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```
bash
# Install all dependencies
pip install -r requirements.txt

# For API development
pip install -r api/requirements.txt
```

## 🎯 Usage

### Option 1: Streamlit Web App

```
bash
# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: FastAPI Backend

```
bash
# Navigate to API directory
cd api

# Run the API server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API root |
| `/health` | GET | Health check |
| `/predict` | POST | Single review prediction |
| `/predict/batch` | POST | Batch review prediction |
| `/predict/csv` | POST | CSV file prediction |

#### Example API Request

```
bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "BEST PRODUCT EVER! Buy now!", "model_type": "logistic_regression"}'
```

### Option 3: Retrain Models

```
bash
# Train new models with updated data
python model/train_model.py
```

## 📊 Model Performance

The models are trained on ~130,000 product reviews with the following performance:

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | ~92% | ~89% | ~85% | ~87% | ~96% |
| Naive Bayes | ~88% | ~82% | ~80% | ~81% | ~94% |
| Linear SVM | ~91% | ~87% | ~84% | ~85% | ~95% |
| Random Forest | ~90% | ~86% | ~83% | ~84% | ~94% |

## 🔍 Detection Features

The system analyzes multiple aspects of reviews:

- **Excessive Exclamations**: 3+ exclamation marks
- **ALL CAPS Text**: Unnatural capitalization
- **Promotional Language**: Urgency tactics, fake guarantees
- **Short Reviews**: Unrealistically brief reviews
- **Repetitive Patterns**: Repeated words or phrases
- **Star Symbols**: Generic star-only reviews
- **Sentiment Analysis**: Extreme positive/negative language

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set requirements.txt path
4. Deploy

### Heroku

```
bash
# Install Heroku CLI
heroku create fake-review-detector
git push heroku main
```

### Docker

```
dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## 📝 API Examples

### Python Client

```
python
import requests

url = "http://localhost:8000/predict"
data = {
    "text": "This product is amazing! Must buy now! Limited offer!",
    "model_type": "logistic_regression"
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript

```
javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "BEST PRODUCT EVER! You must buy now!",
    model_type: "logistic_regression"
  })
});
const data = await response.json();
console.log(data);
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: Amazon Product Reviews
- Built with: Streamlit, FastAPI, Scikit-learn
- Inspired by: Fake Review Detection research papers

---

Made with ❤️ for detecting fake reviews!
