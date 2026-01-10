import { useState } from 'react';
import './App.css';

const API_URL = 'http://127.0.0.1:5000/predict';

// Sentiment Configuration
const SENTIMENT_CONFIG = {
  Positive: { emoji: '😄', bg: 'bg-emerald-500', text: 'text-emerald-600', light: 'bg-emerald-50' },
  Negative: { emoji: '😞', bg: 'bg-rose-500', text: 'text-rose-600', light: 'bg-rose-50' },
  Neutral:  { emoji: '😐', bg: 'bg-slate-400', text: 'text-slate-600', light: 'bg-slate-50' },
};

function App() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeSentiment = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      });

      if (!response.ok) throw new Error('Server error');
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Could not connect to the API. Ensure server.py is running.');
    } finally {
      setLoading(false);
    }
  };

  const config = result ? SENTIMENT_CONFIG[result.sentiment] : null;

  return (
    <div className="min-h-screen flex items-center justify-center p-4 font-sans">
      <div className="w-full max-w-md">
        {/* Card Container */}
        <div className={`rounded-3xl shadow-2xl overflow-hidden transition-colors duration-500 ${config ? config.light : 'bg-white'}`}>
          
          {/* Header */}
          <div className="bg-slate-900 text-white p-6 text-center">
            <h1 className="text-2xl font-bold tracking-tight">SentiMeter</h1>
            <p className="text-slate-400 text-sm mt-1">Hybrid AI · TF-IDF + Word2Vec</p>
          </div>

          {/* Result Display */}
          <div className="p-8 text-center min-h-[200px] flex flex-col items-center justify-center">
            {loading && (
              <div className="animate-pulse">
                <div className="w-16 h-16 rounded-full bg-slate-200 mx-auto mb-4"></div>
                <div className="h-6 w-32 bg-slate-200 rounded mx-auto"></div>
              </div>
            )}

            {error && (
              <div className="text-rose-500">
                <span className="text-4xl mb-2 block">⚠️</span>
                <p className="text-sm">{error}</p>
              </div>
            )}

            {result && !loading && (
              <>
                <span className="text-7xl mb-4 block animate-bounce">{config.emoji}</span>
                <span className={`text-3xl font-extrabold uppercase tracking-wide ${config.text}`}>
                  {result.sentiment}
                </span>
                
                {/* Confidence Bar */}
                <div className="w-full mt-6">
                  <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${config.bg} transition-all duration-700 ease-out`}
                      style={{ width: `${Math.round(result.confidence * 100)}%` }}
                    ></div>
                  </div>
                  <p className="text-slate-500 text-xs mt-2">
                    Confidence: {Math.round(result.confidence * 100)}%
                  </p>
                </div>
              </>
            )}

            {!result && !loading && !error && (
              <div className="text-slate-400">
                <span className="text-5xl mb-3 block">🤖</span>
                <p>Enter a review to analyze</p>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-6 bg-white border-t border-slate-100">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g., The camera is amazing but the battery drains fast..."
              className="w-full p-4 bg-slate-50 rounded-xl border-0 focus:ring-2 focus:ring-slate-900 resize-none h-24 text-sm placeholder:text-slate-400"
            />
            <button
              onClick={analyzeSentiment}
              disabled={loading || !input.trim()}
              className="w-full mt-4 py-3 px-6 bg-slate-900 text-white font-semibold rounded-xl hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? 'Analyzing...' : 'Analyze Sentiment'}
            </button>
          </div>

        </div>

        {/* Footer */}
        <p className="text-center text-slate-400 text-xs mt-6">
          Powered by Word2Vec + TF-IDF Ensemble
        </p>
      </div>
    </div>
  );
}

export default App;
