from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import json
import uuid
import time
import os

from main import TrainingConversation, generate_report_card
from prompts import get_conversation_starter

from document_loader import DocumentProcessor

load_dotenv(dotenv_path='.env')

app = Flask(__name__)
CORS(app, supports_credentials=True)

# INITIALIZE DOCUMENT PROCESSOR (GLOBAL)
doc_processor = DocumentProcessor()

# LOAD COMPANY DOCS AT STARTUP
doc_processor.add_documents_to_vectorstore("company_docs")

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

    convo = TrainingConversation(persona, company_context, vectorstore=doc_processor)

    convo_id = str(uuid.uuid4())
    conversations[convo_id] = convo

    scenario = company_context.get("scenario_context", "")
    initial_prompt = get_conversation_starter(scenario)

    result = convo.send_message(initial_prompt, speak=True)

    return jsonify({
        "convo_id": convo_id,
        "persona": persona["name"],
        "first_message": result["text"],
        "audio_url": f"{result['audio_url']}?t={int(time.time())}" if result["audio_url"] else None
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

    result = convo.send_message(user_message, speak=True)

    return jsonify({
        "response": result["text"],
        "audio_url": f"{result['audio_url']}?t={int(time.time())}" if result["audio_url"] else None
    })


# ------------------------
# AUDIO ROUTE (CRITICAL)
# ------------------------
@app.route('/audio', methods=['GET'])
def get_audio():
    try:
        return send_file("speech.mp3", mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

    report = None
    if convo.get_exchange_count() > 0:
        report = generate_report_card(convo.llm, conversation_data)

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

# ------------------------
# UPLOAD DOCUMENTS (RAG)
# ------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Ensure directory exists
    upload_folder = "company_docs"
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    try:
        # Re-index documents after upload
        doc_processor.add_documents_to_vectorstore(upload_folder)

        return jsonify({
            "message": f"{file.filename} uploaded and indexed successfully"
        })

    except Exception as e:
        return jsonify({
            "error": f"File uploaded but indexing failed: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)