#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║         ServiceSim — Customer Service Trainer        ║
║         Practice interactions & get AI feedback      ║
╚══════════════════════════════════════════════════════╝

Usage:
    python cs_training_bot.py

Requirements:
    pip install anthropic

Set your API key via environment variable:
    export ANTHROPIC_API_KEY="sk-ant-..."
Or the script will prompt you for it.
"""

import os
import sys
import json
import textwrap
import datetime

# ── Try to import anthropic, guide the user if missing ──────────────────────
try:
    import anthropic
except ImportError:
    print("\n[ERROR] The 'anthropic' package is not installed.")
    print("Run:  pip install anthropic\n")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
#  PERSONAS
# ════════════════════════════════════════════════════════════════════════════

PERSONAS = {
    "1": {
        "key": "college_student",
        "name": "College Student",
        "difficulty": "Easy",
        "difficulty_stars": "★☆☆",
        "description": "Young, less knowledgeable, eager to learn",
        "emoji": "🎓",
        "system_prompt": (
            "You are roleplaying as a college student in your early 20s. You're interested in the "
            "product/service but don't have deep knowledge about it. You ask basic questions and are "
            "generally enthusiastic and trusting. You might use casual language and are easily impressed "
            "by features. You have a limited budget and care about getting good value. "
            "Stay in character throughout the conversation."
        ),
    },
    "2": {
        "key": "confused_elderly",
        "name": "Confused Elderly Customer",
        "difficulty": "Medium",
        "difficulty_stars": "★★☆",
        "description": "Not tech-savvy, needs patient explanations",
        "emoji": "👴",
        "system_prompt": (
            "You are roleplaying as an elderly person in your 70s who is not very tech-savvy. You're "
            "polite and well-meaning but easily confused by technical jargon or complex instructions. "
            "You need things explained slowly and simply, possibly multiple times. You might go off on "
            "tangents or reference how things 'used to be.' You appreciate patience and get flustered "
            "if rushed. You're trying your best but need extra help. "
            "Stay in character throughout the conversation."
        ),
    },
    "3": {
        "key": "business_executive",
        "name": "Business Executive",
        "difficulty": "Hard",
        "difficulty_stars": "★★★",
        "description": "Highly knowledgeable, demanding, hard to impress",
        "emoji": "💼",
        "system_prompt": (
            "You are roleplaying as a senior business executive with 20+ years of experience. You are "
            "very knowledgeable about the industry and have high standards. You ask tough, specific "
            "questions and expect data-driven answers. You're skeptical of marketing claims and want "
            "concrete ROI justification. You're polite but direct, and you don't have time for fluff. "
            "You've seen every pitch before. Stay in character throughout the conversation."
        ),
    },
    "4": {
        "key": "cynical_middle_aged",
        "name": "Cynical Middle-Aged Person",
        "difficulty": "Hard",
        "difficulty_stars": "★★★",
        "description": "Hard to convince, skeptical, seen it all before",
        "emoji": "🙄",
        "system_prompt": (
            "You are roleplaying as a middle-aged person in your 40s-50s who has been burned by products "
            "or services before. You're skeptical of claims and need convincing proof. You frequently "
            "bring up past negative experiences and ask 'what makes this different?' You're not hostile, "
            "just deeply skeptical and hard to win over. You want guarantees and are suspicious of "
            "anything that sounds too good to be true. Stay in character throughout the conversation."
        ),
    },
    "5": {
        "key": "upset_customer",
        "name": "Upset Customer",
        "difficulty": "Hard",
        "difficulty_stars": "★★★",
        "description": "Angry about a problem, demanding resolution",
        "emoji": "😤",
        "system_prompt": (
            "You are roleplaying as an upset customer who has had a bad experience with the product or "
            "service. You're frustrated, possibly angry, and feel like your time has been wasted. You "
            "want immediate action and solutions, not excuses. You might interrupt or be slightly "
            "confrontational, but you're not abusive. You'll calm down if you feel genuinely heard and "
            "see real effort to fix the problem. Stay in character throughout the conversation."
        ),
    },
}

# ════════════════════════════════════════════════════════════════════════════
#  COACHING PROMPT  (generates the report card)
# ════════════════════════════════════════════════════════════════════════════

COACHING_PROMPT = """\
You are an expert customer service coach evaluating a training session.
A trainee (AGENT) just practised handling a simulated customer ({persona_name}, difficulty: {difficulty}).

Provide a detailed, constructive report card with EXACTLY this structure and these headings:

OVERALL SCORE
  Give a score X/10 with one sentence justification.

STRENGTHS  (list 3)
  For each, give a title and 1-2 sentence explanation with a specific example from the conversation.

AREAS FOR IMPROVEMENT  (list 3)
  For each, give a title and 1-2 sentence explanation with a specific example from the conversation.

HANDLING THE {persona_upper} PERSONA
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
#  TERMINAL HELPERS
# ════════════════════════════════════════════════════════════════════════════

WIDTH = 70

def hr(char="─"):
    print(char * WIDTH)

def banner(title: str, char="═"):
    print()
    print(char * WIDTH)
    pad = (WIDTH - len(title) - 2) // 2
    print(char * pad + f" {title} " + char * (WIDTH - pad - len(title) - 2))
    print(char * WIDTH)

def section(title: str):
    print()
    print(f"  ┌{'─' * (len(title) + 2)}┐")
    print(f"  │ {title} │")
    print(f"  └{'─' * (len(title) + 2)}┘")

def wrap_print(text: str, indent: int = 4):
    """Word-wrap and print text with an indent."""
    for line in text.splitlines():
        if line.strip() == "":
            print()
        else:
            for wrapped in textwrap.wrap(line, width=WIDTH - indent) or [""]:
                print(" " * indent + wrapped)

def color(text: str, code: str) -> str:
    """ANSI colour helper (gracefully degrades if terminal doesn't support it)."""
    codes = {
        "gold":   "\033[93m",
        "green":  "\033[92m",
        "red":    "\033[91m",
        "cyan":   "\033[96m",
        "bold":   "\033[1m",
        "dim":    "\033[2m",
        "reset":  "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"

# ════════════════════════════════════════════════════════════════════════════
#  API HELPERS
# ════════════════════════════════════════════════════════════════════════════

def make_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print(color("\n  No ANTHROPIC_API_KEY environment variable found.", "red"))
        api_key = input("  Paste your Anthropic API key: ").strip()
        if not api_key:
            print("  No key provided. Exiting.")
            sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def chat(client: anthropic.Anthropic, system: str, messages: list[dict]) -> str:
    """Send messages and return assistant text."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def generate_report(client: anthropic.Anthropic, persona: dict, conversation: list[dict]) -> str:
    """Generate the full report card from the conversation."""
    convo_text = "\n\n".join(
        f"{'AGENT' if m['role'] == 'user' else 'CUSTOMER'}: {m['content']}"
        for m in conversation
    )
    prompt = COACHING_PROMPT.format(
        persona_name=persona["name"],
        difficulty=persona["difficulty"],
        persona_upper=persona["name"].upper(),
    )
    return chat(
        client,
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Here is the training conversation to evaluate:\n\n"
                    f"---\n{convo_text}\n---\n\n"
                    "Please produce the full report card now."
                ),
            }
        ],
    )

# ════════════════════════════════════════════════════════════════════════════
#  SCREENS
# ════════════════════════════════════════════════════════════════════════════

def show_welcome():
    banner("ServiceSim — Customer Service Trainer")
    print()
    wrap_print(
        "Welcome! This tool lets you practise real-world customer service "
        "interactions against AI-powered customer personas. At the end of each "
        "session you'll receive a detailed report card.",
        indent=2,
    )
    print()
    hr()


def show_persona_menu() -> dict:
    section("Select a Customer Persona")
    print()
    for num, p in PERSONAS.items():
        diff_color = {"Easy": "green", "Medium": "gold", "Hard": "red"}[p["difficulty"]]
        print(
            f"  [{color(num, 'bold')}]  {p['emoji']}  "
            f"{color(p['name'], 'bold')}  "
            f"{color(p['difficulty_stars'] + ' ' + p['difficulty'], diff_color)}"
        )
        print(f"       {color(p['description'], 'dim')}")
        print()

    while True:
        choice = input(color("  Choose persona (1-5): ", "cyan")).strip()
        if choice in PERSONAS:
            return PERSONAS[choice]
        print(color("  Invalid choice. Please enter a number 1–5.", "red"))


def show_chat_tips(persona: dict):
    tips = {
        "college_student":   ["Be friendly & approachable", "Highlight value for money", "Keep explanations simple"],
        "confused_elderly":  ["Speak slowly & avoid jargon", "Repeat key points patiently", "Offer reassurance"],
        "business_executive":["Lead with data & ROI",        "Be concise and direct",       "Anticipate hard questions"],
        "cynical_middle_aged":["Acknowledge past bad experiences","Offer solid guarantees",  "Use evidence not hype"],
        "upset_customer":    ["Listen first, solve second",  "Apologise sincerely",         "Give concrete next steps"],
    }
    hints = tips.get(persona["key"], [])
    print()
    print(color("  💡 Coaching Tips for this persona:", "gold"))
    for h in hints:
        print(f"     • {h}")
    print()
    hr("─")


def run_chat(client: anthropic.Anthropic, persona: dict) -> list[dict]:
    """Run the interactive chat loop. Returns the message history."""
    banner(f"{persona['emoji']}  {persona['name']}")
    show_chat_tips(persona)

    print(color("  Type your messages as the customer service AGENT.", "cyan"))
    print(color("  Commands:  'done' → end session & get report   'quit' → exit without report", "dim"))
    print()
    hr("─")

    # Opening line from the customer
    opening_msg = chat(
        client,
        system=persona["system_prompt"],
        messages=[{"role": "user", "content": "Start the conversation naturally as this customer persona. Say a brief opening sentence."}],
    )

    print()
    print(f"  {persona['emoji']}  {color(persona['name'], 'bold')}:")
    wrap_print(opening_msg, indent=6)
    print()

    # We track two separate lists:
    #   api_messages  – sent to the customer persona LLM  (excludes the seed prompt)
    #   history       – the real conversation for the report card
    api_messages: list[dict] = [
        {"role": "user",      "content": "Start the conversation naturally as this customer persona. Say a brief opening sentence."},
        {"role": "assistant", "content": opening_msg},
    ]
    history: list[dict] = [
        {"role": "assistant", "content": opening_msg},
    ]

    while True:
        try:
            user_input = input(color("  🎧 You (Agent): ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Session interrupted.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print(color("\n  Session ended without report. Goodbye!\n", "dim"))
            sys.exit(0)

        if user_input.lower() == "done":
            break

        history.append({"role": "user", "content": user_input})
        api_messages.append({"role": "user", "content": user_input})

        print(color("\n  [Thinking…]\n", "dim"), end="", flush=True)
        reply = chat(client, persona["system_prompt"], api_messages)

        api_messages.append({"role": "assistant", "content": reply})
        history.append({"role": "assistant", "content": reply})

        print(f"  {persona['emoji']}  {color(persona['name'], 'bold')}:")
        wrap_print(reply, indent=6)
        print()
        hr("─")

    return history


def show_report(persona: dict, report_text: str):
    banner("📊  REPORT CARD")
    print()
    print(color(f"  Persona   : {persona['emoji']} {persona['name']}", "bold"))
    print(color(f"  Difficulty: {persona['difficulty_stars']} {persona['difficulty']}", "bold"))
    print(color(f"  Date      : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}", "dim"))
    print()
    hr("═")
    print()
    wrap_print(report_text, indent=2)
    print()
    hr("═")


def offer_save(persona: dict, history: list[dict], report_text: str):
    print()
    choice = input(color("  Save report card to a text file? (y/n): ", "cyan")).strip().lower()
    if choice != "y":
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{persona['key']}_{timestamp}.txt"

    convo_text = "\n\n".join(
        f"{'AGENT' if m['role'] == 'user' else 'CUSTOMER'}: {m['content']}"
        for m in history
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * WIDTH + "\n")
        f.write(f" ServiceSim Report Card\n")
        f.write(f" Persona   : {persona['name']}\n")
        f.write(f" Difficulty: {persona['difficulty']}\n")
        f.write(f" Date      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
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

    print(color(f"\n  ✅  Report saved to: {filename}", "green"))


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    show_welcome()

    client = make_client()

    while True:
        persona = show_persona_menu()

        history = run_chat(client, persona)

        if len(history) < 2:
            print(color("\n  Not enough messages to generate a report. At least one exchange is needed.\n", "red"))
        else:
            print()
            print(color("  ⏳  Generating your report card — this may take a moment…", "gold"))
            report_text = generate_report(client, persona, history)
            show_report(persona, report_text)
            offer_save(persona, history, report_text)

        print()
        again = input(color("  Start another session? (y/n): ", "cyan")).strip().lower()
        if again != "y":
            print(color("\n  Thanks for training with ServiceSim. Good luck! 👋\n", "green"))
            break


if __name__ == "__main__":
    main()
