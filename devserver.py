import os
import datetime
import numpy as np
from flask import Flask, request, jsonify
from transformers import Conversation, pipeline  # Import transformers module
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app

class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    @staticmethod
    def text_to_text(input_text):
        nlp = pipeline("conversational", model="microsoft/DialoGPT-medium")
        chat = nlp(Conversation(input_text), pad_token_id=50256)
        response = str(chat)
        response = response[response.find("bot >> ") + 6:].strip()
        return response

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

@app.route("/request", methods=["POST", "OPTIONS"])
def handle_request():
    if request.method == "OPTIONS":
        # Handle preflight request
        response = app.make_default_options_response()
    else:
        data = request.get_json()
        query = data.get("query")
        if query:
            ai = ChatBot(name="dev")
            if ai.wake_up(query):
                res = "Hello, I am your AI assistant. How can I help you?"
            elif "time" in query:
                res = ai.action_time()
            elif any(i in query for i in ["thank", "thanks"]):
                res = np.random.choice(["You're welcome!", "Anytime!", "No problem!", "Cool!", "I'm here if you need me!", "Mention not"])
            elif any(i in query for i in ["exit", "close"]):
                res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "Peace out!"])
            else:
                res = ai.text_to_text(query)
            response = jsonify({"response": res})
        else:
            response = jsonify({"error": "No query provided"})

    # Set CORS headers
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
