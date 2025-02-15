import os
import random
import json
import pickle
import numpy as np
import nltk
import logging
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from datasets import load_dataset

nltk.download('punkt')
nltk.download('wordnet')
logging.basicConfig(level=logging.INFO)

# Load primary intents from a JSON file or fallback to defaults
def load_intents(file_path="intents.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        logging.error("intents.json not found. Using default intents.")
        return {
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

# Load additional intents from the Hugging Face 'empathetic_dialogues' dataset
def load_hf_intents(num_entries=100):
    try:
        dataset = load_dataset("empathetic_dialogues", split="train")
    except Exception as e:
        logging.error("Error loading Hugging Face dataset: %s", e)
        return {"intents": []}
    
    hf_intents = {"intents": []}
    for i, example in enumerate(dataset):
        if i >= num_entries:
            break
        intent_tag = f"hf_intent_{i}"
        pattern = example.get("utterance", "I feel something")
        response = f"I understand that you said: '{pattern}'. Can you tell me more about that?"
        hf_intents["intents"].append({
            "tag": intent_tag,
            "patterns": [pattern],
            "responses": [response]
        })
    logging.info("Loaded %d Hugging Face intent entries.", len(hf_intents["intents"]))
    return hf_intents

# Merge primary and additional intents
def merge_intents(primary_intents, additional_intents):
    merged = {"intents": primary_intents.get("intents", []).copy()}
    for intent in additional_intents.get("intents", []):
        merged["intents"].append(intent)
    return merged

# Main training function
def train_model():
    lemmatizer = WordNetLemmatizer()

    # Load and merge intents
    primary_intents = load_intents("intents.json")
    hf_intents = load_hf_intents(num_entries=100)
    intents = merge_intents(primary_intents, hf_intents)
    
    # Save merged intents for use by the app
    with open("intents_merged.json", "w") as f:
        json.dump(intents, f)
    logging.info("Merged intents saved to intents_merged.json with %d intents.", len(intents["intents"]))
    
    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ".", ","]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
        for w in words:
            bag.append(1 if w in word_patterns else 0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    
    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # Build and compile the model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation="softmax"))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Save model and preprocessing data
    model.save("chatbot_model.h5")
    with open("words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open("classes.pkl", "wb") as f:
        pickle.dump(classes, f)
    
    return "Training complete. Model updated."

if __name__ == "__main__":
    message = train_model()
    print(message)
