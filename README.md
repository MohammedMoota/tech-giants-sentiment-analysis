# Apple vs. Samsung: Sentiment Analysis

## 📌 Overview

This project compares **Syntactic (TF-IDF)** and **Semantic (Word2Vec)** approaches to classify sentiment in ~14,000 mobile reviews.

**Goal:** Determine if public sentiment favors Apple or Samsung using Machine Learning.

## 📂 Dataset

[**Kaggle: Mobile Reviews Sentiment & Specification**](https://www.kaggle.com/datasets/mohankrishnathalla/mobile-reviews-sentiment-and-specification)

- **Size:** 14,000 Reviews (Balanced).
- **Classes:** Positive, Negative, Neutral.

## ⚙️ How It Works

1. **Preprocessing:** Cleaned text (emojis -> text, fix contractions, lemmatization).
2. **Model A (Syntax):** SVM + TF-IDF. High accuracy (**95%**) by catching specific keywords.
3. **Model B (Semantics):** SVM + Word2Vec. Lower accuracy (**72%**) due to limited data size for embeddings.

## 🚀 Usage

1. Clone repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn nltk gensim`
3. Run `Apple_vs_Samsung_Sentiment_Analysis.ipynb`.

## 👤 Author

**MohammedMoota**
