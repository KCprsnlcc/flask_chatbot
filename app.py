import os
import random
import pickle
import numpy as np
import logging
import json
import nltk
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Utility: Load model and preprocessing files (words, classes)
def load_resources():
    global model, words, classes
    if os.path.exists("chatbot_model.h5"):
        model = load_model("chatbot_model.h5")
        logging.info("Trained model loaded successfully.")
    else:
        model = None
        logging.error("Trained model not found.")
    
    if os.path.exists("words.pkl"):
        with open("words.pkl", "rb") as f:
            words = pickle.load(f)
        logging.info("Vocabulary loaded successfully.")
    else:
        words = []
        logging.error("Vocabulary file not found.")
    
    if os.path.exists("classes.pkl"):
        with open("classes.pkl", "rb") as f:
            classes = pickle.load(f)
        logging.info("Classes loaded successfully.")
    else:
        classes = []
        logging.error("Classes file not found.")

load_resources()

# Utility: Load merged intents (if available) or fallback to a default set
def load_intents():
    file_path = "intents_merged.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            intents = json.load(f)
        logging.info("Loaded merged intents from %s", file_path)
    else:
        logging.error("%s not found. Using default intents.", file_path)
        intents = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hello", "Hi", "Hey", "Good day"],
                    "responses": [
                        "Hello! How are you feeling today?",
                        "Hi there! I'm here to listen.",
                        "Hey! How can I help you today?"
                    ]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["Bye", "See you later", "Goodbye"],
                    "responses": [
                        "Take care!",
                        "Goodbye, remember you're not alone.",
                        "See you later! Stay safe."
                    ]
                },
                {
                    "tag": "mental_health",
                    "patterns": [
                        "I feel stressed",
                        "I'm feeling anxious",
                        "I feel depressed",
                        "I'm overwhelmed"
                    ],
                    "responses": [
                        "I'm really sorry you're feeling this way. It might help to talk to someone you trust or a mental health professional.",
                        "It sounds like you're going through a tough time. Remember, it's okay to ask for help."
                    ]
                }
            ]
        }
    return intents

intents = load_intents()

# Preprocessing: Tokenize and lemmatize a sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

# Convert a sentence into a bag-of-words vector
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent tag for an input sentence
def predict_class(sentence, threshold=0.3):  # lowered threshold from 0.5 to 0.3
    if model is None:
        logging.error("Model not loaded.")
        return "noanswer"
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_prob = np.max(res)
    logging.info("Prediction probabilities: %s", res)
    if max_prob < threshold:
        return "noanswer"
    return classes[np.argmax(res)]


# Home route renders the chatbot interface
@app.route("/")
def home():
    return render_template("index.html")

# Process user message and return an appropriate chatbot response
@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"response": "Invalid request."}), 400
    user_message = data["message"]
    intent_tag = predict_class(user_message)
    if intent_tag == "noanswer":
        response = random.choice([
            "I'm not sure I understand.",
            "Could you rephrase that?",
            "I don't quite get it."
        ])
    else:
        for i in intents["intents"]:
            if i["tag"] == intent_tag:
                response = random.choice(i["responses"])
                break
        else:
            response = "Sorry, something went wrong."
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
