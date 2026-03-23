 ## 📱 SMS Spam Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.9.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) and three different classification algorithms. The best model — a Support Vector Machine — achieves **98% accuracy**.

<img width="819" height="353" alt="image" src="https://github.com/user-attachments/assets/ccb1fea8-856c-44a3-9c27-acb92095d607" />

<img width="1024" height="1536" alt="spam ml banner" src="https://github.com/user-attachments/assets/3af77bfc-d310-4b1b-b18a-b4fa575eab59" />

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Models & Results](#models--results)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## Overview

SMS spam is a pervasive problem. This project builds a text classification pipeline that:

1. Explores and visualizes the SMS dataset
2. Preprocesses raw text using NLP techniques
3. Extracts features via TF-IDF vectorization
4. Trains and compares three ML classifiers
5. Evaluates the best model with a confusion matrix and detailed metrics

---

## Dataset

The project uses the **SMS Spam Collection Dataset** (`spam_sms.csv`).

| Property | Value |
|---|---|
| Total messages | 5,572 |
| Ham messages | 4,825 (86.6%) |
| Spam messages | 747 (13.4%) |
| Features used | `label`, `message` |

> **Source:** [UCI Machine Learning Repository – SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

## Project Pipeline

```
Raw SMS Text
     │
     ▼
┌─────────────────────────┐
│  1. Data Exploration    │  → Class distribution, shape, info
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  2. Text Preprocessing  │  → Lowercase → Remove punctuation
│                         │    → Tokenize → Remove stopwords
│                         │    → Lemmatize
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  3. Feature Extraction  │  → TF-IDF Vectorizer (top 5,000 features)
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  4. Model Training      │  → Naive Bayes
│                         │    → Logistic Regression
│                         │    → Support Vector Machine (SVM)
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  5. Evaluation          │  → Accuracy, Precision, Recall, F1
│                         │    → Confusion Matrix
└─────────────────────────┘
```

---

## Models & Results

The dataset was split **80% training / 20% testing** with stratified sampling.

| Model | Accuracy | Ham F1 | Spam F1 |
|---|---|---|---|
| Naive Bayes | 96.77% | 0.98 | 0.86 |
| Logistic Regression | 95.70% | 0.98 | 0.81 |
| **Support Vector Machine** | **97.85%** | **0.99** | **0.91** |

### 🏆 Best Model: Support Vector Machine (SVM)

```
              precision    recall  f1-score   support

         Ham       0.98      1.00      0.99       966
        Spam       1.00      0.84      0.91       149

    accuracy                           0.98      1115
```

The SVM model achieves **100% precision on spam** — it never incorrectly flags a legitimate message as spam.

---

## Visualizations

The project generates the following plots:

- **Class Distribution** — Bar chart of Ham vs. Spam counts
- **Confusion Matrix** — Heatmap of SVM predictions vs. true labels
- **Word Clouds** — Most frequent words in Ham and Spam messages
- **Top 10 Most Frequent Words** — Bar chart across all messages
- **Message Length Distribution** — Histogram and boxplot by class
- **N-gram Analysis** — Most common word combinations per class

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection

# 2. Install dependencies
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud

# 3. Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

---

## Usage

1. Place `spam_sms.csv` in the project root directory.

2. Run the main script:

```bash
python spam_detection.py
```

Or open and run the notebook cell by cell in **Google Colab** or **Jupyter Notebook**.

### Preprocessing Example

```
Original:  "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
Cleaned:   "go jurong point crazy available bugis n great world la e buffet"
```

---

## Project Structure

```
sms-spam-detection/
│
├── spam_sms.csv              # Dataset
├── spam_detection.py         # Main script (or .ipynb notebook)
└── README.md                 # Project documentation
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.2 | Data loading and manipulation |
| `numpy` | 2.0.2 | Numerical operations |
| `nltk` | 3.9.1 | Text preprocessing (tokenization, stopwords, lemmatization) |
| `scikit-learn` | 1.6.1 | TF-IDF, model training, evaluation |
| `matplotlib` | 3.10.0 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `wordcloud` | 1.9.4 | Word cloud generation |

---

## Future Work

- [ ] Experiment with deep learning models (LSTM, BERT)
- [ ] Add a web interface for real-time spam prediction
- [ ] Explore additional feature engineering (message length, special character count)
- [ ] Handle class imbalance with SMOTE or class weighting
- [ ] Deploy model as a REST API

---

## 👩‍💻 Author

**Meghana Kamatam
M.S. Data Science**

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
