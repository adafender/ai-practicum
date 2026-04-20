from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json
import uuid

from main import TrainingConversation, generate_report_card
from prompts import get_conversation_starter  # 🔥 important

load_dotenv(dotenv_path='.env')

app = Flask(__name__)
CORS(app)

# Load personas
with open('personas.json', 'r') as f:
    personas = json.load(f)

# In-memory store
conversations = {}

# ------------------------
# START CONVERSATION
# ------------------------
@app.route('/start', methods=['POST'])
def start_conversation():
    data = request.json

    if not data or "persona_id" not in data or "company_context" not in data:
        return jsonify({"error": "Invalid request body"}), 400

    persona_id = data["persona_id"]
    company_context = data["company_context"]

    if persona_id not in personas:
        return jsonify({"error": "Invalid persona_id"}), 400

    persona = personas[persona_id]

    convo = TrainingConversation(persona, company_context)

    convo_id = str(uuid.uuid4())
    conversations[convo_id] = convo

    # 🔥 START CONVERSATION WITH REAL SCENARIO CONTEXT
    scenario = company_context.get("scenario_context", "")
    initial_prompt = get_conversation_starter(scenario)

    first_response = convo.send_message(initial_prompt, speak=False)

    return jsonify({
        "convo_id": convo_id,
        "persona": persona["name"],
        "first_message": first_response
    })

# ------------------------
# SEND MESSAGE
# ------------------------
@app.route('/message', methods=['POST'])
def message():
    data = request.json

    if not data or "convo_id" not in data or "message" not in data:
        return jsonify({"error": "Invalid request body"}), 400

    convo_id = data["convo_id"]
    user_message = data["message"]

    convo = conversations.get(convo_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404

    response = convo.send_message(user_message, speak=False)

    return jsonify({
        "response": response
    })

# ------------------------
# GET TRANSCRIPT
# ------------------------
@app.route('/transcript/<convo_id>', methods=['GET'])
def get_transcript(convo_id):
    convo = conversations.get(convo_id)

    if not convo:
        return jsonify({"error": "Conversation not found"}), 404

    return jsonify({
        "transcript": convo.get_transcript()
    })

# ------------------------
# END CONVERSATION
# ------------------------
@app.route('/end', methods=['POST'])
def end_conversation():
    data = request.json

    if not data or "convo_id" not in data:
        return jsonify({"error": "Invalid request body"}), 400

    convo_id = data["convo_id"]

    convo = conversations.get(convo_id)
    if not convo:
        return jsonify({"error": "Conversation not found"}), 404

    conversation_data = convo.get_conversation_data()

    filename = convo.save()

    # 🔥 Generate report card
    report = None
    if convo.get_exchange_count() > 0:
        report = generate_report_card(convo.llm, conversation_data)

    # Remove from memory
    del conversations[convo_id]

    return jsonify({
        "message": "Conversation ended",
        "filename": filename,
        "conversation_data": conversation_data,
        "report_card": report
    })

# ------------------------
# GET PERSONAS
# ------------------------
@app.route('/personas', methods=['GET'])
def get_personas():
    return jsonify(personas)

# ------------------------
# HEALTH CHECK
# ------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

if __name__ == '__main__':
    app.run(debug=True)