import json
import os
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from openai import OpenAI
from dotenv import load_dotenv


# ════════════════════════════════════════════════════════════════════════════
#  REPORT CARD PROMPT
# ════════════════════════════════════════════════════════════════════════════

COACHING_SYSTEM = """\
You are an expert customer service coach evaluating a training session.
A trainee (AGENT) just practised handling a simulated customer.
Provide a detailed, constructive report card with EXACTLY this structure:

OVERALL SCORE
  Give a score X/10 with one sentence justification.

STRENGTHS  (list 3)
  For each, give a title and 1-2 sentence explanation with a specific example from the conversation.

AREAS FOR IMPROVEMENT  (list 3)
  For each, give a title and 1-2 sentence explanation with a specific example from the conversation.

HANDLING THE PERSONA
  2-3 sentences on how well the agent adapted to THIS specific customer type and what they should do differently.

SKILLS BREAKDOWN
  Rate each of the following 1-5 with a one-line comment:
    • Empathy & Active Listening
    • Problem-Solving
    • Clarity of Communication
    • Professionalism & Tone
    • Closing & Next Steps

KEY TAKEAWAY
  One memorable sentence the trainee should carry into their next session.

Be honest, specific, and encouraging. Reference actual lines from the conversation.
"""


# ════════════════════════════════════════════════════════════════════════════
#  REPORT CARD HELPERS
# ════════════════════════════════════════════════════════════════════════════

def generate_report_card(llm, conversation_data):
    """Generate a coaching report card from a completed conversation."""
    persona    = conversation_data["persona"]
    context    = conversation_data["company_context"]
    transcript = conversation_data["transcript"]

    convo_text = "\n\n".join(
        f"{'AGENT' if m['role'] == 'user' else 'CUSTOMER'}: {m['content']}"
        for m in transcript
    )

    user_content = (
        f"Persona    : {persona['name']} (difficulty: {persona['difficulty']})\n"
        f"Traits     : {', '.join(persona.get('traits', []))}\n"
        f"Product    : {context.get('company_product', 'N/A')}\n"
        f"Industry   : {context.get('company_industry', 'N/A')}\n"
        f"Scenario   : {context.get('scenario_context', 'N/A')}\n"
        f"Exchanges  : {conversation_data['exchange_count']}\n\n"
        f"Conversation transcript:\n---\n{convo_text}\n---\n\n"
        "Please produce the full report card now."
    )

    messages = [
        SystemMessage(content=COACHING_SYSTEM),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    return response.content


def print_report_card(report_text, conversation_data):
    """Pretty-print the report card to the terminal."""
    WIDTH    = 70
    persona  = conversation_data["persona"]
    duration = int(conversation_data["duration_seconds"])

    print()
    print("═" * WIDTH)
    title = "  REPORT CARD"
    pad   = (WIDTH - len(title) - 2) // 2
    print("═" * pad + f" {title} " + "═" * (WIDTH - pad - len(title) - 2))
    print("═" * WIDTH)
    print()
    print(f"  Persona   : {persona['name']}")
    print(f"  Difficulty: {persona['difficulty']}")
    print(f"  Exchanges : {conversation_data['exchange_count']}")
    print(f"  Duration  : {duration // 60}m {duration % 60}s")
    print(f"  Date      : {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print()
    print("─" * WIDTH)
    print()

    for line in report_text.splitlines():
        if line.strip() == "":
            print()
        else:
            for wrapped in textwrap.wrap(line, width=WIDTH - 2) or [""]:
                print("  " + wrapped)

    print()
    print("═" * WIDTH)


def save_report_card(report_text, conversation_data, filename=None):
    """Save the report card as a plain-text file."""
    if filename is None:
        ts       = datetime.fromisoformat(conversation_data["timestamp"]).strftime("%Y%m%d_%H%M%S")
        filename = f"report_{ts}.txt"

    WIDTH    = 70
    persona  = conversation_data["persona"]
    duration = int(conversation_data["duration_seconds"])

    convo_text = "\n\n".join(
        f"{'AGENT' if m['role'] == 'user' else 'CUSTOMER'}: {m['content']}"
        for m in conversation_data["transcript"]
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * WIDTH + "\n")
        f.write(" ServiceSim Report Card\n")
        f.write(f" Persona   : {persona['name']}\n")
        f.write(f" Difficulty: {persona['difficulty']}\n")
        f.write(f" Exchanges : {conversation_data['exchange_count']}\n")
        f.write(f" Duration  : {duration // 60}m {duration % 60}s\n")
        f.write(f" Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * WIDTH + "\n\n")
        f.write("CONVERSATION TRANSCRIPT\n")
        f.write("-" * WIDTH + "\n\n")
        f.write(convo_text)
        f.write("\n\n")
        f.write("=" * WIDTH + "\n")
        f.write("AI COACH REPORT\n")
        f.write("=" * WIDTH + "\n\n")
        f.write(report_text)
        f.write("\n")

    return filename


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING CONVERSATION CLASS
# ════════════════════════════════════════════════════════════════════════════

class TrainingConversation:
    def __init__(self, persona, company_context, model="gpt-4o-mini", tts_voice="shimmer"):
        """
        Initialize a training conversation with a specific persona using LangChain.

        Args:
            persona: Persona dict from personas.json
            company_context: Dict with company_product, scenario_context, company_industry
            model: OpenAI model to use
            tts_voice: Voice for text-to-speech (shimmer, alloy, echo, fable, onyx, nova)
        """
        load_dotenv()

        self.persona         = persona
        self.company_context = company_context
        self.start_time      = datetime.now()
        self.tts_voice       = tts_voice

        # Speech output file (reused each turn)
        self.speech_file_path = Path(__file__).parent / "speech.mp3"

        # Initialize LangChain + OpenAI clients
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Memory stores conversation history
        self.chat_history = ChatMessageHistory()

        # Create system prompt
        self.system_prompt = self._create_system_prompt()
        self.messages = [SystemMessage(content=self.system_prompt)]

    def _create_system_prompt(self):
        """Create the system prompt from persona template."""
        return self.persona["system_prompt_template"].format(
            age_range=self.persona["demographics"]["age_range"],
            **self.company_context
        )

    def _speak(self, text):
        """Convert text to speech using OpenAI TTS and play it."""
        try:
            with self.openai_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=self.tts_voice,
                input=text,
                instructions="Speak naturally as a customer in a sales/support conversation.",
            ) as response:
                response.stream_to_file(self.speech_file_path)

            self._play_audio(self.speech_file_path)

        except Exception as e:
            print(f"[TTS error: {e}]")

    def _play_audio(self, file_path):
        """Play an audio file using the system's default player."""
        import platform
        system = platform.system()

        if system == "Darwin":        # macOS
            subprocess.run(["afplay", str(file_path)], check=True)
        elif system == "Linux":
            subprocess.run(["mpg123", str(file_path)], check=True)
        elif system == "Windows":
            os.startfile(str(file_path))
        else:
            print(f"[Audio playback not supported on {system} — file saved to {file_path}]")

    def send_message(self, user_message, speak=True):
        """
        Send a message, get the AI response, and optionally speak it aloud.

        Args:
            user_message: String message from the trainee
            speak: Whether to read the response aloud (default True)

        Returns:
            String response from the AI persona
        """
        messages = [self.messages[0]]  # System message
        messages.extend(self.chat_history.messages)

        user_msg = HumanMessage(content=user_message)
        messages.append(user_msg)

        response = self.llm.invoke(messages)

        self.chat_history.add_user_message(user_message)
        self.chat_history.add_ai_message(response.content)
        self.messages.append(user_msg)
        self.messages.append(AIMessage(content=response.content))

        if speak:
            self._speak(response.content)

        return response.content

    def get_transcript(self):
        """Get conversation transcript without system prompt."""
        transcript = []
        for msg in self.messages[1:]:
            if isinstance(msg, HumanMessage):
                transcript.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                transcript.append({"role": "assistant", "content": msg.content})
        return transcript

    def get_exchange_count(self):
        """Get number of back-and-forth exchanges."""
        return len(self.get_transcript()) // 2

    def get_conversation_data(self):
        """Get all conversation data for saving/evaluation."""
        return {
            "timestamp": self.start_time.isoformat(),
            "persona": {
                "id":         self.persona["persona_id"],
                "name":       self.persona["name"],
                "difficulty": self.persona["personality"]["difficulty"],
                "traits":     self.persona["personality"]["traits"]
            },
            "company_context":  self.company_context,
            "transcript":       self.get_transcript(),
            "exchange_count":   self.get_exchange_count(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }

    def save(self, filename=None):
        """Save conversation to JSON file."""
        if filename is None:
            filename = f"conversation_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(self.get_conversation_data(), f, indent=2)

        return filename

    def clear_history(self):
        """Clear conversation history (useful for testing)."""
        self.chat_history.clear()
        self.messages = [self.messages[0]]


# ════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE SESSION
# ════════════════════════════════════════════════════════════════════════════

def run_interactive_session():
    """Run an interactive training session in the terminal."""
    load_dotenv()

    with open('personas.json', 'r') as f:
        personas = json.load(f)

    print("=== Available Personas ===")
    for i, (key, persona) in enumerate(personas.items(), 1):
        print(f"{i}. {persona['name']} ({persona['personality']['difficulty']})")

    choice       = int(input("\nSelect persona number: ")) - 1
    selected_key = list(personas.keys())[choice]
    persona      = personas[selected_key]

    company_context = {
        "company_product":  input("Product/Service: "),
        "scenario_context": input("Scenario context: "),
        "company_industry": input("Industry: ")
    }

    # Voice selection
    voices = ["shimmer", "alloy", "echo", "fable", "onyx", "nova"]
    print("\nAvailable voices: " + ", ".join(voices))
    voice_input = input("Choose a voice (default: shimmer): ").strip().lower()
    tts_voice   = voice_input if voice_input in voices else "shimmer"

    conversation = TrainingConversation(persona, company_context, tts_voice=tts_voice)

    print(f"\n{'='*50}")
    print(f"Persona   : {persona['name']}")
    print(f"Difficulty: {persona['personality']['difficulty']}")
    print(f"Voice     : {tts_voice}")
    print(f"{'='*50}\n")
    print("Type 'quit' to end | 'mute' to silence speech | 'unmute' to restore it\n")

    speaking = True

    while True:
        user_input = input("You: ").strip()

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

        response = conversation.send_message(user_input, speak=speaking)
        print(f"\nCustomer: {response}\n")

    # ── Save conversation JSON ────────────────────────────────────────────
    conv_filename     = conversation.save()
    conversation_data = conversation.get_conversation_data()

    print(f"\n{'='*50}")
    print("Conversation Complete!")
    print(f"Total exchanges: {conversation.get_exchange_count()}")
    print(f"Saved to: {conv_filename}")
    print(f"{'='*50}")

    # ── Generate & display report card ───────────────────────────────────
    if conversation.get_exchange_count() < 1:
        print("\n[No exchanges recorded — skipping report card.]\n")
        return

    print("\nGenerating your report card, please wait…")
    report_text = generate_report_card(conversation.llm, conversation_data)

    print_report_card(report_text, conversation_data)

    # Offer to save as text file
    save_choice = input("\nSave report card to a text file? (y/n): ").strip().lower()
    if save_choice == "y":
        report_filename = save_report_card(report_text, conversation_data)
        print(f"Report saved to: {report_filename}")

    # Embed report inside the conversation JSON
    conversation_data["report_card"] = report_text
    with open(conv_filename, "w") as f:
        json.dump(conversation_data, f, indent=2)
    print(f"Report also embedded in: {conv_filename}")


if __name__ == "__main__":
    run_interactive_session()