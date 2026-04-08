"""
Next Word Prediction - Flask Backend
Loads a pre-trained LSTM model and tokenizer,
exposes a /predict endpoint that returns top 3 next words.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__, static_folder=".")
CORS(app)

model = None
tokenizer = None
MAX_SEQ_LEN = None


# =========================
# LOAD MODEL + TOKENIZER
# =========================
def load_model_and_tokenizer():
    global model, tokenizer, MAX_SEQ_LEN

    try:
        from tensorflow.keras.models import load_model
        import pickle

        base_path = os.path.dirname(__file__)

        model_path = os.path.join(base_path, "sugg_next_word_v3.h5")
        tokenizer_path = os.path.join(base_path, "tokenizer.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        # Load model
        model = load_model(model_path)

        # Get sequence length from model
        MAX_SEQ_LEN = model.input_shape[1]

        # Load tokenizer
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        print("[INFO] Model loaded successfully")
        print("[INFO] Tokenizer loaded successfully")
        print(f"[INFO] Max sequence length: {MAX_SEQ_LEN}")

    except Exception as e:
        print(f"[ERROR] {e}")
        print("[WARN] Running in DEMO mode")


# =========================
# PREDICTION FUNCTION
# =========================
def predict_next_words(text, top_n=3):

    # DEMO MODE (if model not loaded)
    if model is None or tokenizer is None:
        demo_words = [
            "playing", "happy", "running", "going",
            "looking", "thinking", "walking", "reading"
        ]
        return demo_words[:top_n]

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Convert text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=MAX_SEQ_LEN,
        padding='pre'
    )

    # Predict probabilities
    predictions = model.predict(token_list, verbose=0)[0]

    # Convert index → word
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_word[0] = ""

    # Temperature scaling (controls randomness)
    temperature = 0.7
    preds = np.asarray(predictions).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # Get top N words (like mobile keyboard)
    top_indices = preds.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        word = index_to_word.get(idx)
        if word and word not in results:
            results.append(word)

    return results


# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text'"}), 400

    text = str(data["text"]).strip()

    if text == "":
        return jsonify({"predictions": []})

    try:
        words = predict_next_words(text, top_n=3)
        return jsonify({"predictions": words})

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": "Prediction failed"}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    load_model_and_tokenizer()
    app.run(debug=True, port=5000)