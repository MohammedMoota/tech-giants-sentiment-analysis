# Tech Giants Showdown: Apple vs. Samsung Sentiment Analysis

## 📌 Project Overview

This project performs a comparative sentiment analysis on approximately 14,000 mobile reviews for Apple and Samsung devices (2023–2025). It aims to determine public sentiment and evaluate the performance of different machine learning models in classifying reviews as **Positive**, **Negative**, or **Neutral**.

The core analysis compares two approaches:

1. **Syntactic Method**: TF-IDF Vectorization with Support Vector Machine (SVM).
2. **Semantic Method**: Word2Vec Embeddings with Support Vector Machine (SVM).

## 📂 Dataset

- **Source**: Kaggle "Mobile Reviews Sentiment" dataset.
- **Volume**: ~14,000 reviews (Filtered for Apple and Samsung).
- **Balance**: Balanced distribution between Apple (~7,100) and Samsung (~7,000) reviews.
- **Classes**: Positive, Negative, Neutral.

## 🛠️ Methodology & Models

### 1. Data Preprocessing (The Pipeline)

An industrial-strength NLP pipeline was implemented to clean and normalize text:

- **Emoji Translation**: Converted emojis (e.g., 😡, 😍) into sentiment-bearing text.
- **Contraction Expansion**: "won't" → "will not".
- **Noise Removal**: Stripped non-alphabetic characters.
- **Lemmatization**: Reduced words to their base forms using NLTK's WordNetLemmatizer.
- **Duplicate Removal**: Ensured data integrity by removing exact duplicates.

### 2. Model A: SVM + TF-IDF (Syntactic)

- **Feature Engineering**: Uses Term Frequency-Inverse Document Frequency (TF-IDF) to convert text to vectors, focusing on word importance.
- **Algorithm**: Linear SVC with regularization (C=0.3).
- **Strength**: Excellent at capturing keyword-based sentiment.

### 3. Model B: SVM + Word2Vec (Semantic)

- **Feature Engineering**: Uses Gensim's Word2Vec to create dense vector embeddings that capture semantic meaning and context.
- **Algorithm**: Linear SVC with strict regularization (C=0.1).
- **Strength**: Understands context better but often requires larger datasets to outperform TF-IDF on specific domains.

## 📊 Results Summary

| Model | Accuracy | Insight |
|-------|----------|---------|
| **TF-IDF + SVM** | **~95%** | Highly effective for this dataset; specific keywords are strong predictors of sentiment. |
| **Word2Vec + SVM** | **~72%** | Lower accuracy suggests the dataset size may be insufficient to fully leverage dense embeddings, or the keyword signals are simply stronger. |

*Note: The project favors the Syntax-based model (TF-IDF) for this specific application.*

## 🚀 Usage

1. Clone this repository.
2. Ensure you have the required libraries installed:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk gensim
    ```

3. Open the Jupyter Notebook `NLP_Assignement.ipynb` to run the analysis.

## 👤 Author

**MohammedMoota**
