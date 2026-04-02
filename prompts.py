"""
Prompts and templates for the AI training system.
"""

# ════════════════════════════════════════════════════════════════════════════
#  COACHING/EVALUATION PROMPT
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
#  RAG INSTRUCTION
# ════════════════════════════════════════════════════════════════════════════

RAG_INSTRUCTION = """\

You have access to company documents and training materials. 
When relevant information is provided to you from these documents, 
incorporate it naturally into your responses as a customer would 
if they had read the company's materials (website, policies, etc.).\
"""


# ════════════════════════════════════════════════════════════════════════════
#  CONCISE RESPONSE INSTRUCTION
# ════════════════════════════════════════════════════════════════════════════

CONCISE_INSTRUCTION = """\

IMPORTANT: Keep your responses SHORT and conversational (1-3 sentences max). 
Respond like a real person in a phone/chat conversation, not like you're writing an essay. 
Don't over-explain. Let the conversation flow naturally.\
"""


# ════════════════════════════════════════════════════════════════════════════
#  CONVERSATION STARTER TEMPLATE
# ════════════════════════════════════════════════════════════════════════════

def get_conversation_starter(scenario):
    """Generate the initial prompt to start a customer conversation."""
    return f"You are starting a conversation. Scenario: {scenario}. Begin as the customer would with a brief opening statement."


# ════════════════════════════════════════════════════════════════════════════
#  REPORT CARD USER CONTENT TEMPLATE
# ════════════════════════════════════════════════════════════════════════════

def format_evaluation_request(persona, context, transcript, exchange_count):
    """Format the user message for report card generation."""
    convo_text = "\n\n".join(
        f"{'AGENT' if m['role'] == 'user' else 'CUSTOMER'}: {m['content']}"
        for m in transcript
    )
    
    return (
        f"Persona    : {persona['name']} (difficulty: {persona['difficulty']})\n"
        f"Traits     : {', '.join(persona.get('traits', []))}\n"
        f"Product    : {context.get('company_product', 'N/A')}\n"
        f"Industry   : {context.get('company_industry', 'N/A')}\n"
        f"Scenario   : {context.get('scenario_context', 'N/A')}\n"
        f"Exchanges  : {exchange_count}\n\n"
        f"Conversation transcript:\n---\n{convo_text}\n---\n\n"
        "Please produce the full report card now."
    )
