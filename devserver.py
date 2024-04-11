import os
import datetime
import numpy as np
from flask import Flask, request, jsonify
from transformers import Conversation, pipeline

# Initialize Flask app
app = Flask(__name__)

# Set Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    def get_user_input(self):
        self.text = request.json.get("query")

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

@app.route("/", methods=["POST"])
def chatbot():
    ai = ChatBot(name="dev")
    ai.get_user_input()

    if ai.wake_up(ai.text):
        res = "Hello I am Dave the AI, what can I do for you?"
    elif "time" in ai.text:
        res = ai.action_time()
    elif any(i in ai.text for i in ["thank", "thanks"]):
        res = np.random.choice(["you're welcome!", "anytime!", "no problem!", "cool!", "I'm here if you need me!", "mention not"])
    elif any(i in ai.text for i in ["exit", "close"]):
        res = np.random.choice(["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "peace out!"])
    else:
        res = ai.text_to_text(ai.text)

    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
