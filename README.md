# Sinhala YouTube Comment Classification using XLM-RoBERTa

This project classifies YouTube comments (Sinhala, English, and Singlish) into violence-related categories using multilingual NLP techniques. It features a fine-tuned XLM-RoBERTa model, an advanced preprocessing pipeline, and classical model benchmarking.

---

## 🚀 Features

- 🔤 **Multilingual Support**: Sinhala, Singlish (transliterated), and English
- 🧽 **Preprocessing**: Emoji removal, stopword removal, punctuation & number cleanup, language translation & transliteration
- 🤖 **Model**: Fine-tuned **XLM-RoBERTa** transformer for multilingual classification
- ⚙️ **Comparison**: Benchmarked against Logistic Regression, Naive Bayes, Random Forest, SVM, and Decision Tree
- 🔗 **YouTube Integration**: Extracts real comments from any YouTube video
- 📈 **Outputs**: Prediction label with confidence, plus preprocessing trace

---

## 📂 Project Structure

```text
project-root/
│
├── notebooks/                  # Saved XLM-RoBERTa model
│   └── saved_model/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── label_encoder.joblib
│
├── youtube_predictor.py        # Core module: extraction, preprocessing, prediction
├── app.py                      # Flask app interface
├── requirements.txt            # Required Python packages
└── README.md                   # Project description
```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SuwanthaDarshana/youtube_comments_analyse
cd youtube_comments_analyse
```

### 2. (Optional) Create a Virtual Environment

```bash
pip install virtualenv
virtualenv env
env\Scripts\activate  # Windows
# or
source env/bin/activate  # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python app.py
```

Then open your browser and go to:  
👉 [http://localhost:5000](http://localhost:5000)

- Paste a YouTube video URL
- See the predicted category of the comments

---

## 🔧 Technologies Used

### Languages & Frameworks
- Python 3
- Flask
- PyTorch
- Hugging Face Transformers
- Scikit-learn

### Preprocessing Tools
- Googletrans (for translation to Sinhala)
- `re` + `string.punctuation` for text cleanup

### Machine Learning / NLP
- TF-IDF vectorization (for classical models)
- Classifiers: Logistic Regression, Naive Bayes, Random Forest, SVM, Decision Tree
- Fine-tuned XLM-RoBERTa transformer

---

## 💡 Novelty of Our Approach

- 🈳 **Multilingual NLP**: Supports Sinhala, English, and Singlish using transliteration and translation
- 🔀 **Hybrid Evaluation**: Benchmarks classical ML models vs modern transformers
- 🧠 **Transformer Fine-Tuning**: Trained on real-world, code-mixed Sinhala YouTube comments
- 🎯 **Confidence-Driven Output**: Displays the most confident prediction from a batch of comments

---

## 🔮 Future Improvements

- 🚀 Deploy on **Hugging Face Spaces** or package with **Docker** for easier sharing and testing