# Advanced Traditional Chatbot (Non-LLM Based)

## 📌 Overview
This project implements a traditional Natural Language Processing (NLP) chatbot using **TF-IDF vectorization** and **cosine similarity** for intent classification.

Unlike modern Large Language Model (LLM)-based systems, this chatbot uses classical, interpretable techniques for conversational interaction.

The chatbot is built using **Python, Flask, NLTK, and Scikit-learn**, and includes a modern web-based user interface with real-time interaction and confidence scoring.

---

## 🚀 Features

- 50+ predefined intents
- TF-IDF vectorization
- Cosine similarity matching
- Lemmatization-based preprocessing
- Confidence score display
- Modern dark-themed chat UI
- AJAX-based real-time interaction
- Malformed input handling with fallback mechanism
- Extendable architecture (cloud/API ready)

---

## 🛠 Technologies Used

- Python
- Flask
- NLTK
- Scikit-learn
- NumPy
- HTML
- CSS
- JavaScript

---

## 🏗 Project Structure

```
Traditional-Chatbot/
│
├── app.py
├── intents.json
├── requirements.txt
├── README.md
├── screenshots/
│     ├── ui.png
│     ├── greeting.png
│     ├── fallback.png
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Traditional-Chatbot.git
cd Traditional-Chatbot
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate environment:

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## 🧠 How It Works

1. User enters a message.
2. Text is preprocessed (lowercasing, tokenization, lemmatization).
3. TF-IDF transforms the input into a numerical vector.
4. Cosine similarity compares input with predefined intent patterns.
5. The highest similarity intent is selected.
6. Confidence score is calculated.
7. Response is returned to the user interface.

If similarity is below the threshold (0.35), a fallback message is triggered.

---

## 📊 Example Interaction

**User:** What is machine learning?  
**Bot:** Machine learning is a subset of AI that allows systems to learn from data and improve performance without explicit programming.  
**Confidence:** 0.842  

---

## 📈 Future Improvements

- Sentiment analysis integration
- Persistent chat history storage
- Azure Cognitive Services integration
- Hybrid ML classifier
- Cloud deployment

---

## 👨‍💻 Author

Ravi Kiran Raju Kanumuri  
MSAI631 – Artificial Intelligence for Human-Computer Interaction  
University of the Cumberlands  

---

## 📄 License

This project was developed for academic purposes.
