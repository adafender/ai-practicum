import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load personas
with open('personas.json', 'r') as f:
    personas = json.load(f)

# Display menu
print("Select a customer persona:")
for i, (key, persona) in enumerate(personas.items(), 1):
    print(f"{i}. {persona['name']} - {persona['description']} (Difficulty: {persona['difficulty']})")

selected = personas["upset_customer"]  # For testing, we select the upset customer persona directly

print(f"\n{'='*50}")
print(f"Testing: {selected['name']}")
print(f"{'='*50}\n")

messages = [
    {"role": "system", "content": selected["system_prompt"]},
    {"role": "user", "content": "Hello! Thanks for taking the time to speak with me today. I'd love to tell you about our new project management software."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Customer response:")
print(response.choices[0].message.content)