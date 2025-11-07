🧠 Project Title:

“AI-based Medical Chatbot using Machine Learning (Decision Tree + Neural Network)”

🎯 Project Overview

This project is an AI-powered medical chatbot that can answer basic disease-related queries from users.
It uses Machine Learning to predict the most relevant cure based on the given disease name or symptom description.

⚙️ Working Process (Simple Flow)

Here’s how it works — step by step:

1️⃣ Dataset Preparation

The dataset you used (disease_cure_dataset.csv) contains two columns:

Disease / Symptom → (Input from user)

Cure / Treatment → (Expected output from chatbot)

Example:

Disease	Cure
Fever	Take paracetamol, rest, and drink fluids.
Diabetes	Maintain diet, exercise, monitor sugar levels.
2️⃣ Text Preprocessing

Before training, all text is cleaned and converted into numerical form so the model can understand it.

Steps:

Convert all text to lowercase

Remove punctuation/special characters

Convert words into vectors using TF-IDF Vectorization (Term Frequency–Inverse Document Frequency)

📘 TF-IDF basically measures how important a word is in a sentence compared to the whole dataset —
so “fever” or “cough” will get higher weight than common words like “the”, “is”.

3️⃣ Algorithms Used

Your project actually uses two machine learning models for prediction:

🧩 1. Decision Tree Classifier

Purpose: To learn simple rule-based mappings between diseases and cures.

How it works:
It splits data into “if–else” conditions internally, e.g.

if text contains "fever" → answer = "Take paracetamol"
else if text contains "cold" → answer = "Take rest"


It’s fast and interpretable but can overfit on small data.

🧠 2. MLPClassifier (Multi-Layer Perceptron Neural Network)

Purpose: To generalize better for unseen questions.

How it works:
It’s a feed-forward neural network with multiple layers that learns complex patterns from the text embeddings (vectorized data).
It captures semantic similarity, so even if someone types
“I have high temperature” → it can still predict the “fever” cure.

You used layers like (64, 32) neurons for compact performance.

✅ Your code actually trains both models, and then tries:

Neural Network (MLP) first

Decision Tree as backup

If both fail, uses Cosine Similarity (vector-based text matching) to find the most similar question from the dataset.

4️⃣ Response Generation

When a user types:

“I have cough and sore throat”

The chatbot:

Cleans and vectorizes it

Passes it to the trained models

Model predicts → “Drink warm fluids, rest, and take cough syrup.”

Flask sends this answer back as a JSON response to the frontend.

5️⃣ Frontend & Backend Communication

Frontend: index.html + style.css
(Chat interface built using HTML, CSS, and JavaScript)

Backend: Flask

/ask API route handles user messages

Calls get_best_answer() → returns AI response

Everything runs seamlessly either locally (127.0.0.1:5000) or live (Render app).

🧮 Architecture Summary (In One Line)

“User → Flask API → Text Preprocessing → TF-IDF → Machine Learning Models (Decision Tree + MLP Neural Network) → Predicted Cure”

📈 Advantages

✅ Works offline (if models are pre-trained)
✅ Lightweight — only needs CSV + ML models
✅ Can be expanded with new medical Q&A data
✅ Easy deployment on Render or Replit

⚠️ Limitations

It’s not a real medical diagnostic system (uses static dataset, not live hospital data).

For serious medical use, it should be integrated with verified datasets or APIs (like WHO or CDC data).
