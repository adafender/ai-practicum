import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load personas
with open('personas.json', 'r') as f:
    personas = json.load(f)

# Test data
company_context = {
    "company_product": "CRM software",
    "scenario_context": "managing customer relationships for their sales team",
    "company_industry": "SaaS"
}

# Pick a persona
persona = personas["gen_z_budget_student"]

# Fill in template
system_prompt = persona["system_prompt_template"].format(
    age_range=persona["demographics"]["age_range"],
    **company_context
)

print(f"Testing: {persona['name']}\n")

# Have a conversation
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Hi! I'd love to tell you about our CRM software."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)