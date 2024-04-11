import os
import datetime
import numpy as np
from flask import Flask, request, jsonify
from transformers import Conversation, pipeline
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app

# Set Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    @staticmethod
    def text_to_text(input_text):
        nlp = pipeline("conversational", model="microsoft/DialoGPT-medium", token=HF_TOKEN)
        chat = nlp(Conversation(input_text), pad_token_id=50256)
        response = str(chat)
        response = response[response.find("bot >> ") + 6:].strip()
        return response

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

@app.route("/query", methods=["POST"])
def handle_query():
    query = request.json.get("query")
    if query:
        ai = ChatBot(name="dev")
        if ai.wake_up(query):
            res = "Hello I am Dave the AI, what can I do for you?"
        elif "time" in query:
            res = ai.action_time()
        elif any(i in query for i in ["thank", "thanks"]):
            res = np.random.choice(["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "mention not"])
        elif any(i in query for i in ["exit", "close"]):
            res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "peace out!"])
        else:
            res = ai.text_to_text(query)
        response = jsonify({"response": res})
        response.headers.add("Access-Control-Allow-Origin", "*")  # Set the CORS header for all origins
        return response
    else:
        return jsonify({"error": "No query provided"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
