
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import joblib
import os

# --- CONFIGURATION ---
DATA_FILE = 'Mobile Reviews Sentiment.csv'
MODEL_FILE = 'models.joblib'
RANDOM_STATE = 42

# --- 1. SETUP & RESOURCES ---
def download_nltk_resources():
    """Download necessary NLTK data if not present."""
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            print(f"Downloading {res}...")
            nltk.download(res, quiet=True)

# --- 2. CLEANING PIPELINE ---
def clean_text(text):
    """
    Standardizes text: lowercases, removes garbage, handles contractions,
    fixes elongations, handles negations, removes stopwords, and lemmatizes.
    """
    if not isinstance(text, str): return ""
    text = text.lower()
    
    # Basic Cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+', '', text)
    
    # Contraction Expansion
    contractions = {
        r"won't": "will not", r"can't": "cannot", r"n't": " not",
        r"'re": " are", r"'s": " is", r"'d": " would",
        r"'ll": " will", r"'t": " not", r"'ve": " have", r"'m": " am"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # Elongation Fix (e.g., "goood" -> "good")
    text = re.sub(r'(.)\1+', r'\1\1', text)

    # Negation Handling
    negation_words = {"not", "no", "never", "none", "hardly", "scarcely", "barely", "don't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't"}
    words = text.split()
    new_words = []
    i = 0
    while i < len(words):
        word = words[i]
        if word in negation_words:
            new_words.append(word)
            j = i + 1
            while j < len(words) and words[j].isalpha():
                new_words.append(words[j] + "_NOT")
                j += 1
            i = j
        else:
            new_words.append(word)
            i += 1
    text = " ".join(new_words)

    # Stopwords & Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])
    
    return text

def get_word2vec_embedding(tokens, model):
    """Calculates the average vector for a list of tokens."""
    if not tokens: return np.zeros(model.vector_size)
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not valid_vectors: return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

# --- 3. MAIN TRAINING FLOW ---
def main():
    download_nltk_resources()
    
    print(f"1. Loading Data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print("Error: Dataset not found.")
        return

    df = pd.read_csv(DATA_FILE)
    # Filter for relevant brands
    df = df[df['brand'].isin(['Apple', 'Samsung'])].copy()
    
    print("2. Preprocessing Data...")
    df['clean_text'] = df['review_text'].apply(clean_text)
    y = df['sentiment'].values # Labels: Positive, Negative, Neutral

    # --- MODEL A: SYNTACTIC (TF-IDF + SVM) ---
    print("\n3. Training Model A (Syntactic)...")
    tfidf = TfidfVectorizer(max_features=15, stop_words='english')
    X_tfidf = tfidf.fit_transform(df['clean_text']).toarray()
    
    model_a = SVC(kernel='linear', C=0.3, probability=True, random_state=RANDOM_STATE)
    model_a.fit(X_tfidf, y)
    print("   Model A Trained.")

    # --- MODEL B: SEMANTIC (Word2Vec + SVM) ---
    print("\n4. Training Model B (Semantic)...")
    tokenized_text = [t.split() for t in df['clean_text']]
    
    w2v_model = Word2Vec(sentences=tokenized_text, vector_size=32, window=3, min_count=1, seed=RANDOM_STATE, workers=1)
    
    X_w2v = np.array([get_word2vec_embedding(t, w2v_model) for t in tokenized_text])
    scaler = MinMaxScaler()
    X_w2v_scaled = scaler.fit_transform(X_w2v)
    
    model_b = SVC(kernel='linear', C=0.1, probability=True, random_state=RANDOM_STATE)
    model_b.fit(X_w2v_scaled, y)
    print("   Model B Trained.")

    # --- SAVE ARTIFACTS ---
    print(f"\n5. Saving Hybrid Artifacts to {MODEL_FILE}...")
    artifacts = {
        'model_a': model_a,
        'tfidf': tfidf,
        'model_b': model_b,
        'w2v': w2v_model,
        'scaler': scaler
    }
    joblib.dump(artifacts, MODEL_FILE)
    print("   Success! All models saved in one file.")

if __name__ == "__main__":
    main()
