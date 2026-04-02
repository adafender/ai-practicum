"""
Configuration constants for the AI training system.
"""

# ════════════════════════════════════════════════════════════════════════════
#  MODEL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TEMPERATURE = 0.7

# ════════════════════════════════════════════════════════════════════════════
#  TTS CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

AVAILABLE_VOICES = ["shimmer", "alloy", "echo", "fable", "onyx", "nova"]
DEFAULT_VOICE = "shimmer"

# ════════════════════════════════════════════════════════════════════════════
#  RAG CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_RETRIEVAL_K = 3  # Number of document chunks to retrieve

# ════════════════════════════════════════════════════════════════════════════
#  REPORT CARD FORMATTING
# ════════════════════════════════════════════════════════════════════════════

REPORT_WIDTH = 70

# ════════════════════════════════════════════════════════════════════════════
#  FILE PATHS
# ════════════════════════════════════════════════════════════════════════════

PERSONAS_FILE = "personas.json"
SPEECH_FILE = "speech.mp3"

# ════════════════════════════════════════════════════════════════════════════
#  EXAMPLE SCENARIOS
# ════════════════════════════════════════════════════════════════════════════

EXAMPLE_SCENARIOS = [
    "A customer calling about a delayed shipment for an online clothing store",
    "A potential client asking about pricing for our cloud storage service",
    "An upset customer whose credit card was charged twice",
    "A new user needing help setting up their software account"
]
