"""
AI-Powered Customer Interaction Training Platform
Main conversation system with RAG, TTS, and automated evaluation.
"""

import json
import os
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import OpenAI
from dotenv import load_dotenv

from prompts import (
    COACHING_SYSTEM, 
    RAG_INSTRUCTION, 
    CONCISE_INSTRUCTION,
    get_conversation_starter,
    format_evaluation_request
)
from config import (
    DEFAULT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TEMPERATURE,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    DEFAULT_RETRIEVAL_K,
    REPORT_WIDTH,
    PERSONAS_FILE,
    SPEECH_FILE,
    EXAMPLE_SCENARIOS
)


# ════════════════════════════════════════════════════════════════════════════
#  REPORT CARD HELPERS
# ════════════════════════════════════════════════════════════════════════════

def generate_report_card(llm, conversation_data, retriever=None):
    """Generate a coaching report card from a completed conversation."""
    persona = conversation_data["persona"]
    base_context = conversation_data.get("company_context", "")
    transcript = conversation_data["transcript"]
    exchange_count = conversation_data["exchange_count"]
    
    # Convert transcript to string for retrieval
    transcript_text = "\n".join([m["content"] for m in transcript])
    
    # Retrieve relevant documents using RAG if retriever is available
    rag_context = ""
    if retriever:
        try:
            results = retriever.retrieve(transcript_text, top_k=4)
            if results:
                rag_context = "\n\n".join([
                    f"[Company Doc]:\n{doc.page_content}" for doc in results
                ])
        except Exception as e:
            print(f"[Report RAG error: {e}]")
            rag_context = ""
    
    # Smart fallback logic (handles ALL cases)
    if rag_context and base_context:
        context = f"{base_context}\n\nRelevant Policies:\n{rag_context}"
    elif rag_context:
        context = rag_context
    elif base_context:
        context = base_context
    else:
        context = "No company guidelines provided. Give general sales coaching."
    
    # Build evaluation request (unchanged)
    user_content = format_evaluation_request(persona, context, transcript, exchange_count)
    
    messages = [
        SystemMessage(content=COACHING_SYSTEM),
        HumanMessage(content=user_content),
    ]
    
    response = llm.invoke(messages)
    return response.content


def print_report_card(report_text, conversation_data):
    """Pretty-print the report card to the terminal."""
    persona = conversation_data["persona"]
    duration = int(conversation_data["duration_seconds"])
    
    print()
    print("═" * REPORT_WIDTH)
    title = "  REPORT CARD"
    pad = (REPORT_WIDTH - len(title) - 2) // 2
    print("═" * pad + f" {title} " + "═" * (REPORT_WIDTH - pad - len(title) - 2))
    print("═" * REPORT_WIDTH)
    print()
    print(f"  Persona   : {persona['name']}")
    print(f"  Difficulty: {persona['difficulty']}")
    print(f"  Exchanges : {conversation_data['exchange_count']}")
    print(f"  Duration  : {duration // 60}m {duration % 60}s")
    print(f"  Date      : {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print()
    print("─" * REPORT_WIDTH)
    print()
    
    for line in report_text.splitlines():
        if line.strip() == "":
            print()
        else:
            for wrapped in textwrap.wrap(line, width=REPORT_WIDTH - 2) or [""]:
                print("  " + wrapped)
    
    print()
    print("═" * REPORT_WIDTH)


def save_report_card(report_text, conversation_data, filename=None):
    """Save the report card as a plain-text file."""
    if filename is None:
        ts = datetime.fromisoformat(conversation_data["timestamp"]).strftime("%Y%m%d_%H%M%S")
        filename = f"report_{ts}.txt"
    
    persona = conversation_data["persona"]
    duration = int(conversation_data["duration_seconds"])
    
    convo_text = "\n\n".join(
        f"{'AGENT' if m['role'] == 'user' else 'CUSTOMER'}: {m['content']}"
        for m in conversation_data["transcript"]
    )
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * REPORT_WIDTH + "\n")
        f.write(" AI Training Platform Report Card\n")
        f.write(f" Persona   : {persona['name']}\n")
        f.write(f" Difficulty: {persona['difficulty']}\n")
        f.write(f" Exchanges : {conversation_data['exchange_count']}\n")
        f.write(f" Duration  : {duration // 60}m {duration % 60}s\n")
        f.write(f" Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * REPORT_WIDTH + "\n\n")
        f.write("CONVERSATION TRANSCRIPT\n")
        f.write("-" * REPORT_WIDTH + "\n\n")
        f.write(convo_text)
        f.write("\n\n")
        f.write("=" * REPORT_WIDTH + "\n")
        f.write("AI COACH REPORT\n")
        f.write("=" * REPORT_WIDTH + "\n\n")
        f.write(report_text)
        f.write("\n")
    
    return filename


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING CONVERSATION CLASS
# ════════════════════════════════════════════════════════════════════════════

class TrainingConversation:
    def __init__(self, persona, company_context, vectorstore=None, 
                 model=DEFAULT_MODEL, tts_voice=DEFAULT_VOICE):
        """
        Initialize a training conversation with RAG, TTS, and report card capabilities.
        
        Args:
            persona: Persona dict from personas.json
            company_context: Dict with company_product, scenario_context, company_industry
            vectorstore: Optional vectorstore for RAG (DocumentProcessor instance)
            model: OpenAI model to use
            tts_voice: Voice for text-to-speech
        """
        load_dotenv()
        
        self.persona = persona
        self.company_context = company_context
        self.vectorstore = vectorstore
        self.start_time = datetime.now()
        self.tts_voice = tts_voice
        
        # Speech output file (reused each turn)
        self.speech_file_path = Path(__file__).parent / SPEECH_FILE
        
        # Initialize LangChain + OpenAI clients
        self.llm = ChatOpenAI(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple in-memory chat history
        self.chat_history = []
        
        # Create system prompt
        self.system_prompt = self._create_system_prompt()
        self.messages = [SystemMessage(content=self.system_prompt)]
    
    def _create_system_prompt(self):
        """Create the system prompt from persona template."""
        base_prompt = self.persona["system_prompt_template"].format(
            age_range=self.persona["demographics"]["age_range"],
            **self.company_context
        )
        
        # Add RAG instruction if vectorstore exists
        if self.vectorstore:
            base_prompt += RAG_INSTRUCTION
        
        # Add instruction for concise responses
        base_prompt += CONCISE_INSTRUCTION
        
        return base_prompt
    
    def _retrieve_context(self, user_message, k=DEFAULT_RETRIEVAL_K):
        """Retrieve relevant context from vectorstore."""
        if not self.vectorstore:
            return ""
        
        try:
            results = self.vectorstore.retrieve(user_message, top_k=k)
            if not results:
                return ""
            context = "\n\n".join([
                f"[Company Document Excerpt]:\n{doc.page_content}"
                for doc in results
            ])
            return context
        except Exception as e:
            print(f"[RAG retrieval error: {e}]")
            return ""
    
    def _speak(self, text):
        """Convert text to speech and save it (no local playback)."""
        try:
            with self.openai_client.audio.speech.with_streaming_response.create(
                model=DEFAULT_TTS_MODEL,
                voice=self.tts_voice,
                input=text,
                instructions="Speak naturally as a customer in a sales/support conversation.",
            ) as response:
                response.stream_to_file(self.speech_file_path)
            
            return str(self.speech_file_path)  

        except Exception as e:
            print(f"[TTS error: {e}]")
            return None
    
    def _play_audio(self, file_path):
        """Play an audio file using the system's default player."""
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["afplay", str(file_path)], check=True)
            elif system == "Linux":
                subprocess.run(["mpg123", str(file_path)], check=True)
            elif system == "Windows":
                os.startfile(str(file_path))
            else:
                print(f"[Audio playback not supported on {system}]")
        except Exception as e:
            print(f"[Audio playback error: {e}]")
    
    def send_message(self, user_message, speak=True):
        """
        Send a message and get the AI response with RAG and optional TTS.

        Returns:
            dict: { "text": ..., "audio_url": ... }
        """
        context = self._retrieve_context(user_message)

        messages = [self.messages[0]]

        # Add chat history
        for role, content in self.chat_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        # STRONGER RAG INJECTION 
        if context:
            messages.append(SystemMessage(content=f"""
    You are roleplaying as a CUSTOMER interacting with a company.

    You have access to internal knowledge about this company.

    You MUST use the following company information when responding:
    {context}

    If the representative says something incorrect or contradicts this information,
    you should naturally question or challenge it like a real customer would.

    Do NOT ignore this information.
    """))

        # Add user message
        user_msg = HumanMessage(content=user_message)
        messages.append(user_msg)

        # Get response
        response = self.llm.invoke(messages)

        # Store history
        self.chat_history.append(("user", user_message))
        self.chat_history.append(("assistant", response.content))

        self.messages.append(user_msg)
        self.messages.append(AIMessage(content=response.content))

        print(f"\nCustomer: {response.content}\n")

        audio_url = None

        if speak:
            audio_path = self._speak(response.content)

            if audio_path:
                audio_url = "/audio"

        return {
            "text": response.content,
            "audio_url": audio_url
        }
    
    def get_transcript(self):
        """Get conversation transcript without system prompt."""
        return [{"role": role, "content": content} for role, content in self.chat_history]
    
    def get_exchange_count(self):
        """Get number of back-and-forth exchanges."""
        return len(self.chat_history) // 2
    
    def get_conversation_data(self):
        """Get all conversation data for saving/evaluation."""
        return {
            "timestamp": self.start_time.isoformat(),
            "persona": {
                "id": self.persona["persona_id"],
                "name": self.persona["name"],
                "difficulty": self.persona["personality"]["difficulty"],
                "traits": self.persona["personality"]["traits"]
            },
            "company_context": self.company_context,
            "rag_enabled": self.vectorstore is not None,
            "transcript": self.get_transcript(),
            "exchange_count": self.get_exchange_count(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def save(self, filename=None):
        """Save conversation to JSON file."""
        if filename is None:
            filename = f"conversation_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.get_conversation_data(), f, indent=2)
        
        return filename


# ════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE SESSION
# ════════════════════════════════════════════════════════════════════════════

def run_training_session():
    """Run an interactive training session with RAG, TTS, and report card."""
    from document_loader import DocumentProcessor
    
    load_dotenv()
    
    # Load personas
    with open(PERSONAS_FILE, 'r') as f:
        personas = json.load(f)
    
    # Ask about RAG
    use_rag = input("Use company documents (RAG)? (y/n): ").lower() == 'y'
    
    vectorstore = None
    if use_rag:
        docs_path = input("Path to company documents folder: ")
        processor = DocumentProcessor()
        processor.add_documents_to_vectorstore(docs_path)
        vectorstore = processor
    
    # Display available personas
    print("\n=== Available Personas ===")
    for i, (key, persona) in enumerate(personas.items(), 1):
        print(f"{i}. {persona['name']} ({persona['personality']['difficulty']})")
    
    # Select persona
    choice = int(input("\nSelect persona number: ")) - 1
    selected_key = list(personas.keys())[choice]
    persona = personas[selected_key]
    
    # Get scenario
    print("\nDescribe the scenario in one sentence:")
    for example in EXAMPLE_SCENARIOS:
        print(f"  Example: '{example}'")
    scenario_input = input("\nScenario: ").strip()
    
    company_context = {
        "company_product": scenario_input,
        "scenario_context": scenario_input,
        "company_industry": scenario_input
    }
    
    # Voice selection
    print(f"\nAvailable voices: {', '.join(AVAILABLE_VOICES)}")
    voice_input = input(f"Choose a voice (default: {DEFAULT_VOICE}): ").strip().lower()
    tts_voice = voice_input if voice_input in AVAILABLE_VOICES else DEFAULT_VOICE
    
    # Create conversation
    conversation = TrainingConversation(persona, company_context, vectorstore, tts_voice=tts_voice)
    
    # Display session info
    print(f"\n{'='*50}")
    print(f"Persona   : {persona['name']}")
    print(f"Difficulty: {persona['personality']['difficulty']}")
    print(f"RAG       : {use_rag}")
    print(f"Voice     : {tts_voice}")
    print(f"{'='*50}\n")
    print("YOU are the salesperson/service representative.")
    print("The AI will roleplay as the CUSTOMER.")
    print("The customer will initiate the conversation.\n")
    print("Commands: 'quit' to end | 'mute' to silence | 'unmute' to restore\n")
    
    speaking = True
    
    # Customer starts the conversation
    initial_prompt = get_conversation_starter(scenario_input)
    conversation.send_message(initial_prompt, speak=speaking)
    
    # Conversation loop
    while True:
        user_input = input("You (Rep): ").strip()
        
        if user_input.lower() == 'quit':
            print("\nEnding conversation...")
            break
        elif user_input.lower() == 'mute':
            speaking = False
            print("[Speech muted]\n")
            continue
        elif user_input.lower() == 'unmute':
            speaking = True
            print("[Speech unmuted]\n")
            continue
        
        if user_input:
            conversation.send_message(user_input, speak=speaking)
    
    # Save conversation
    conv_filename = conversation.save()
    conversation_data = conversation.get_conversation_data()
    
    print(f"\n{'='*50}")
    print("Conversation Complete!")
    print(f"Total exchanges: {conversation.get_exchange_count()}")
    print(f"Saved to: {conv_filename}")
    print(f"{'='*50}")
    
    # Generate report card
    if conversation.get_exchange_count() < 1:
        print("\n[No exchanges recorded — skipping report card.]\n")
        return
    
    print("\nGenerating your report card, please wait…")
    report_text = generate_report_card(
    conversation.llm,
    conversation_data,
    retriever=conversation.vectorstore
    )
    
    print_report_card(report_text, conversation_data)
    
    # Save report
    save_choice = input("\nSave report card to a text file? (y/n): ").strip().lower()
    if save_choice == "y":
        report_filename = save_report_card(report_text, conversation_data)
        print(f"Report saved to: {report_filename}")
    
    # Embed report in JSON
    conversation_data["report_card"] = report_text
    with open(conv_filename, "w") as f:
        json.dump(conversation_data, f, indent=2)
    print(f"Report also embedded in: {conv_filename}")


if __name__ == "__main__":
    run_training_session()