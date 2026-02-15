import json
import random
import nltk
import numpy as np
import datetime
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# ------------------------
# TEXT PREPROCESSING
# ------------------------

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return " ".join(tokens)

# ------------------------
# LOAD INTENTS
# ------------------------

with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess(pattern))
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# ------------------------
# RESPONSE LOGIC
# ------------------------

def get_response(user_input):
    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(user_vector, X)

    index = np.argmax(similarity)
    score = float(similarity[0][index])

    if score < 0.35:
        return {
            "response": "I'm not confident I understand that. Could you rephrase?",
            "confidence": round(score, 3)
        }

    tag = tags[index]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return {
                "response": random.choice(intent["responses"]),
                "confidence": round(score, 3)
            }

# ------------------------
# WEB ROUTES
# ------------------------

@app.route("/")
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Traditional Chatbot</title>
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
}

.chat-container {
    width: 450px;
    height: 650px;
    background: #111827;
    border-radius: 20px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    overflow: hidden;
}

.header {
    background: #1f2937;
    padding: 15px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    color: #38bdf8;
}

.chat-box {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}

.message {
    margin: 8px 0;
    padding: 12px 18px;
    border-radius: 18px;
    max-width: 75%;
    animation: fadeIn 0.3s ease-in-out;
}

.user {
    background: #3b82f6;
    color: white;
    align-self: flex-end;
}

.bot {
    background: #374151;
    color: white;
    align-self: flex-start;
}

.input-area {
    display: flex;
    padding: 12px;
    background: #1f2937;
}

input {
    flex: 1;
    padding: 12px;
    border-radius: 25px;
    border: none;
    outline: none;
    font-size: 14px;
}

button {
    margin-left: 10px;
    padding: 12px 18px;
    border-radius: 25px;
    border: none;
    background: #38bdf8;
    color: white;
    cursor: pointer;
    font-weight: bold;
}

button:hover {
    background: #0ea5e9;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
</head>

<body>
<div class="chat-container">
    <div class="header">Traditional Chatbot</div>
    <div class="chat-box" id="chatBox"></div>
    <div class="input-area">
        <input type="text" id="userInput" placeholder="Type your message..." onkeydown="if(event.key==='Enter') sendMessage();">
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
function sendMessage() {
    let inputField = document.getElementById("userInput");
    let message = inputField.value.trim();
    if (message === "") return;

    let chatBox = document.getElementById("chatBox");

    chatBox.innerHTML += `<div class="message user">${message}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<div class="message bot">
        ${data.response}<br><small style="opacity:0.6;">Confidence: ${data.confidence}</small>
        </div>`;

        chatBox.scrollTop = chatBox.scrollHeight;
    });

    inputField.value = "";
}
</script>

</body>
</html>
"""

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    result = get_response(user_input)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
