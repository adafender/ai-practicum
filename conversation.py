import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
from datetime import datetime

class TrainingConversation:
    def __init__(self, persona, company_context, model="gpt-4o-mini"):
        """
        Initialize a training conversation with a specific persona using LangChain.
        
        Args:
            persona: Persona dict from personas.json
            company_context: Dict with company_product, scenario_context, company_industry
            model: OpenAI model to use
        """
        load_dotenv()
        
        self.persona = persona
        self.company_context = company_context
        self.start_time = datetime.now()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,  # Slightly random for more realistic conversations
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Memory stores conversation history
        self.chat_history = ChatMessageHistory()
        
        # Create system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Store messages (for compatibility and saving)
        self.messages = [SystemMessage(content=self.system_prompt)]
    
    def _create_system_prompt(self):
        """Create the system prompt from persona template."""
        return self.persona["system_prompt_template"].format(
            age_range=self.persona["demographics"]["age_range"],
            **self.company_context
        )
    
    def send_message(self, user_message):
        """
        Send a message and get the AI response using LangChain.
        
        Args:
            user_message: String message from the trainee
            
        Returns:
            String response from the AI persona
        """
        # Create message list with system prompt and history
        messages = [self.messages[0]]  # System message
        
        # Add conversation history
        messages.extend(self.chat_history.messages)
        
        # Add current user message
        user_msg = HumanMessage(content=user_message)
        messages.append(user_msg)
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Save to history
        self.chat_history.add_user_message(user_message)
        self.chat_history.add_ai_message(response.content)
        
        # Store in messages for transcript
        self.messages.append(user_msg)
        self.messages.append(AIMessage(content=response.content))
        
        return response.content
    
    def get_transcript(self):
        """Get conversation transcript without system prompt."""
        transcript = []
        for msg in self.messages[1:]:  # Skip system message
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
                "id": self.persona["persona_id"],
                "name": self.persona["name"],
                "difficulty": self.persona["personality"]["difficulty"],
                "traits": self.persona["personality"]["traits"]
            },
            "company_context": self.company_context,
            "transcript": self.get_transcript(),
            "exchange_count": self.get_exchange_count(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def save(self, filename=None):
        """
        Save conversation to JSON file.
        
        Args:
            filename: Optional custom filename, otherwise generates timestamp-based name
        """
        if filename is None:
            filename = f"conversation_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.get_conversation_data(), f, indent=2)
        
        return filename
    
    def clear_history(self):
        """Clear conversation history (useful for testing)."""
        self.chat_history.clear()
        self.messages = [self.messages[0]]  # Keep system prompt


def run_interactive_session():
    """Run an interactive training session in the terminal."""
    # Load personas
    with open('personas.json', 'r') as f:
        personas = json.load(f)
    
    # Display available personas
    print("=== Available Personas ===")
    for i, (key, persona) in enumerate(personas.items(), 1):
        print(f"{i}. {persona['name']} ({persona['personality']['difficulty']})")
    
    # Select persona
    choice = int(input("\nSelect persona number: ")) - 1
    selected_key = list(personas.keys())[choice]
    persona = personas[selected_key]
    
    # Set company context
    company_context = {
        "company_product": input("Product/Service: "),
        "scenario_context": input("Scenario context: "),
        "company_industry": input("Industry: ")
    }
    
    # Create conversation
    conversation = TrainingConversation(persona, company_context)
    
    # Display session info
    print(f"\n{'='*50}")
    print(f"Persona: {persona['name']}")
    print(f"Difficulty: {persona['personality']['difficulty']}")
    print(f"{'='*50}\n")
    print("Type 'quit' to end the conversation\n")
    
    # Conversation loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("\nEnding conversation...")
            break
        
        response = conversation.send_message(user_input)
        print(f"\nCustomer: {response}\n")
    
    # Save and display results
    filename = conversation.save()
    print(f"\n{'='*50}")
    print("Conversation Complete!")
    print(f"Total exchanges: {conversation.get_exchange_count()}")
    print(f"Saved to: {filename}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_interactive_session()