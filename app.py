from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

app = Flask(__name__)

DATASET_PATH = "disease_cure_dataset.csv"  # ✅ exact match
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
DTREE_PATH = os.path.join(MODEL_DIR, "dtree.pkl")
MLP_PATH = os.path.join(MODEL_DIR, "mlp.pkl")

# ---------------- NLP Cleaning ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- Model Setup ----------------
def train_models():
    print("📘 Loading dataset...")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ Dataset not found: {DATASET_PATH}. Place it in the same folder as app.py.")
    
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # ✅ Expecting 'disease' and 'cure' columns
    if "disease" in df.columns and "cure" in df.columns:
        df.rename(columns={"disease": "question", "cure": "answer"}, inplace=True)
    else:
        raise ValueError(f"❌ Expected columns: 'disease' and 'cure'. Found: {df.columns.tolist()}")

    # Clean text
    df["question"] = df["question"].astype(str).apply(clean_text)
    df["answer"] = df["answer"].astype(str)
    df.dropna(subset=["question", "answer"], inplace=True)

    print(f"✅ Loaded {len(df)} rows for training.")

    # Vectorize
    print("🧠 Vectorizing disease names...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["question"])
    y = df["answer"]

    print("🌳 Training Decision Tree...")
    dtree = DecisionTreeClassifier(max_depth=40, random_state=42)
    dtree.fit(X, y)

    print("🧩 Training Neural Network (MLP)...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    mlp.fit(X, y)

    print("💾 Saving models...")
    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(dtree, DTREE_PATH)
    joblib.dump(mlp, MLP_PATH)

    print("✅ Models trained and saved successfully.")
    return vectorizer, dtree, mlp, df


# ---------------- Load or Train ----------------
if os.path.exists(VEC_PATH) and os.path.exists(DATASET_PATH):
    vectorizer = joblib.load(VEC_PATH)
    dtree = joblib.load(DTREE_PATH)
    mlp = joblib.load(MLP_PATH)
    df = pd.read_csv(DATASET_PATH)
    df.rename(columns={"disease": "question", "cure": "answer"}, inplace=True)
    print("✅ Models loaded from disk.")
else:
    vectorizer, dtree, mlp, df = train_models()


# ---------------- Helper Function ----------------
def get_best_answer(user_input):
    user_input_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])

    # 1️⃣ Try Neural Network Prediction
    try:
        mlp_pred = mlp.predict(user_vec)[0]
        return mlp_pred
    except Exception:
        pass

    # 2️⃣ Try Decision Tree Prediction
    try:
        dt_pred = dtree.predict(user_vec)[0]
        return dt_pred
    except Exception:
        pass

    # 3️⃣ Fallback: Cosine Similarity
    X = vectorizer.transform(df["question"].apply(clean_text))
    similarities = cosine_similarity(user_vec, X)
    idx = similarities.argmax()
    return df.iloc[idx]["answer"]


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("message", "")
    answer = get_best_answer(query)
    return jsonify({"reply": answer})


# ---------------- Main ----------------
if __name__ == "__main__":
    print("🚀 Starting Disease–Cure Chatbot (Flask + NLP + ML)...")
    app.run(debug=True)
