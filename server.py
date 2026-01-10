from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for React

# --- 1. SETUP & RESOURCES ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# --- 2. PREPROCESSING (Shared) ---
def clean_text_pipeline(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    
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

    text = re.sub(r'(.)\1+', r'\1\1', text)

    negation_words = ["not", "no", "never", "none", "hardly", "scarcely", "barely", "don't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't"]
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

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def get_word2vec_embedding(tokens, model):
    if not tokens: return np.zeros(model.vector_size)
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not valid_vectors: return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

# --- 3. MODEL LOADING ---
print("Loading Hybrid Models...")
MODELS = {}
try:
    MODELS = joblib.load('models.joblib')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS:
        return jsonify({'error': 'Models not loaded'}), 500
    
    data = request.get_json()
    review = data.get('text', '')
    if not review:
        return jsonify({'error': 'No text provided'}), 400

    # 1. Clean Text
    cleaned_text = clean_text_pipeline(review)
    
    # 2. Model A Prediction (Syntactic - TF-IDF)
    tfidf_vector = MODELS['tfidf'].transform([cleaned_text]).toarray()
    prob_a = MODELS['model_a'].predict_proba(tfidf_vector)[0]
    
    # 3. Model B Prediction (Semantic - Word2Vec)
    tokens = cleaned_text.split()
    w2v_vector = get_word2vec_embedding(tokens, MODELS['w2v'])
    w2v_scaled = MODELS['scaler'].transform(w2v_vector.reshape(1, -1))
    prob_b = MODELS['model_b'].predict_proba(w2v_scaled)[0]
    
    # 4. Ensemble (Soft Voting)
    # Average the probabilities from both models
    final_prob = (prob_a + prob_b) / 2
    
    # Labels: 0=Negative, 1=Neutral, 2=Positive
    labels = ["Negative", "Neutral", "Positive"]
    max_idx = np.argmax(final_prob)
    
    result = {
        'sentiment': labels[max_idx],
        'confidence': float(final_prob[max_idx]),
        'ensemble_breakdown': {
            'model_a_prob': {l: float(p) for l, p in zip(labels, prob_a)},
            'model_b_prob': {l: float(p) for l, p in zip(labels, prob_b)},
            'combined_prob': {l: float(p) for l, p in zip(labels, final_prob)}
        }
    }
    
    return jsonify(result)

if __name__ == '__main__':
    # Disable debug/reloader to reduce memory
    app.run(debug=False, use_reloader=False, port=5000)
