# Sinhala YouTube Comment Classification using XLM-RoBERTa

This project classifies YouTube comments (Sinhala, English, and Singlish) into violence-related categories using multilingual NLP techniques. It features a fine-tuned XLM-RoBERTa model, advanced preprocessing pipeline, and classical model benchmarking.

---

## 🚀 Features

- 🔤 Multilingual support: Sinhala, Singlish (transliterated), and English
- 🧽 Preprocessing: Emoji removal, stopword removal, punctuation & number cleanup, language translation & transliteration
- 🤖 Model: Fine-tuned **XLM-RoBERTa** transformer for multilingual classification
- ⚙️ Comparison: Benchmarked against Logistic Regression, Naive Bayes, Random Forest, SVM, and Decision Tree
- 🔗 YouTube Integration: Extracts real comments from any YouTube video
- 📈 Outputs: Prediction label with confidence, plus preprocessing trace

---

## 📂 Project Structure
project-root/
│
├── notebooks/ # Saved XLM-RoBERTa model
│ └── saved_model/
│ ├── config.json
│ ├── pytorch_model.bin
│ ├── tokenizer.json
│ └── label_encoder.joblib
│
├── youtube_predictor.py # Core module: extraction, preprocessing, prediction
├── app.py # Flask app interface
├── requirements.txt # Required Python packages
└── README.md # This file


---

##Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SuwanthaDarshana/youtube_comments_analyse
cd youtube_comments_analyse


### 2. Set up a virtual environment (optional but recommended)
pip install virtualenv

### 3. Install Dependencies
pip install -r requirements.txt


##  Running the Application
python app.py

Open your browser and visit: http://localhost:5000
Enter a YouTube URL → View predicted category.


🔧 Technologies Used
Languages & Frameworks:

Python 3

Flask

PyTorch

Hugging Face Transformers

Scikit-learn

Preprocessing Tools:

Googletrans (Translation to Sinhala)

Regex & string.punctuation (Cleaning text)

Machine Learning / NLP:

TF-IDF vectorization (for classical models)

Classifiers: Logistic Regression, Naive Bayes, Random Forest, SVM, Decision Tree

Fine-tuned XLM-RoBERTa Transformer

💡 Novelty of Our Approach
Multilingual NLP: Handles Sinhala, English, and Singlish with transliteration & translation

Hybrid Evaluation: Classical models vs modern transformers

Transformer Fine-Tuning: Model trained on real Sinhala YouTube comments for accuracy in noisy, code-mixed environments

Confidence-Driven Output: Returns class with highest confidence after analyzing hundreds of comments

📌 Future Improvements
Deploy on Hugging Face Spaces or Docker