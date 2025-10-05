# 🩺 Healthcare Predictive Diagnosis Bot

A full-stack, conversational chatbot application designed to assist users by providing preliminary disease diagnosis based on reported symptoms using Machine Learning.

The bot utilizes a Finite State Machine (FSM) architecture within its backend to manage multi-turn conversational flow, ensuring accurate collection of patient data before generating a prediction.

## Key Features

* **Predictive Diagnosis:** Uses a supervised machine learning model (Decision Tree Classifier) trained on a large symptom dataset to predict a preliminary disease.
* **Conversational Triage (FSM):** Implements a robust State Machine to guide the user through a series of questions (name, age, main symptom, secondary symptoms) for refined accuracy.
* **Web Interface:** Accessible via a simple, clean, and responsive web interface using HTML/CSS.
* **Text-to-Speech (TTS):** Uses the Web Speech API to automatically read bot responses aloud, with a toggle icon for user control.


Run the Application using: python backend/app.py

Install all necessary dependencies using the provided requirements.txt file