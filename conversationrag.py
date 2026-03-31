# conversation_rag.py
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from datetime import datetime

class TrainingConversationWithRAG:
    def __init__(self, persona, company_context, vectorstore=None, model="gpt-4o-mini"):
        """
        Initialize a training conversation with RAG capabilities.
        """
        load_dotenv()
        
        self.persona = persona
        self.company_context = company_context
        self.vectorstore = vectorstore
        self.start_time = datetime.now()
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Simple in-memory chat history
        self.chat_history = []

        self.system_prompt = self._create_system_prompt()
        self.messages = [SystemMessage(content=self.system_prompt)]
    
    def _create_system_prompt(self):
        """Create the system prompt from persona template."""
        base_prompt = self.persona["system_prompt_template"].format(
            age_range=self.persona["demographics"]["age_range"],
            **self.company_context
        )
        if self.vectorstore:
            rag_instruction = (
                "\n\nYou have access to company documents. "
                "Incorporate relevant information naturally into your responses."
            )
            base_prompt += rag_instruction
        return base_prompt
    
    def _retrieve_context(self, user_message, k=3):
        """Retrieve relevant context from vectorstore."""
        if not self.vectorstore:
            return ""
        results = self.vectorstore.retrieve(user_message, top_k=k)
        if not results:
            return ""
        context = "\n\n".join([f"[Company Document Excerpt]:\n{doc.page_content}" for doc in results])
        return context
    
    def send_message(self, user_message):
        """Send a message and get AI response with RAG."""
        context = self._retrieve_context(user_message)
        
        messages = [self.messages[0]]  # System prompt
        # Add conversation history
        for role, content in self.chat_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        
        if context:
            messages.append(SystemMessage(content=f"Relevant company information:\n{context}"))
        
        messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(messages)
        
        # Update history
        self.chat_history.append(("user", user_message))
        self.chat_history.append(("assistant", response.content))
        self.messages.append(HumanMessage(content=user_message))
        self.messages.append(AIMessage(content=response.content))
        
        return response.content
    
    def get_transcript(self):
        """Get conversation transcript without system prompt."""
        transcript = []
        for role, content in self.chat_history:
            transcript.append({"role": role, "content": content})
        return transcript
    
    def get_exchange_count(self):
        return len(self.chat_history) // 2
    
    def get_conversation_data(self):
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
        if filename is None:
            filename = f"conversation_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.get_conversation_data(), f, indent=2)
        return filename
    
    def clear_history(self):
        self.chat_history = []
        self.messages = [self.messages[0]]


def run_interactive_session_with_rag():
    from document_loader import DocumentProcessor
    
    # Load personas
    with open("personas.json", "r") as f:
        personas = json.load(f)
    
    use_rag = input("Use company documents (RAG)? (y/n): ").lower() == "y"
    
    vectorstore = None
    if use_rag:
        docs_path = input("Path to company documents folder: ")
        processor = DocumentProcessor()
        processor.add_documents_to_vectorstore(docs_path)
        vectorstore = processor  # now has retrieve() method
    
    # Display personas
    print("\n=== Available Personas ===")
    for i, (key, persona) in enumerate(personas.items(), 1):
        print(f"{i}. {persona['name']} ({persona['personality']['difficulty']})")
    
    choice = int(input("\nSelect persona number: ")) - 1
    selected_key = list(personas.keys())[choice]
    persona = personas[selected_key]
    
    company_context = {
        "company_product": input("Product/Service: "),
        "scenario_context": input("Scenario context: "),
        "company_industry": input("Industry: ")
    }
    
    conversation = TrainingConversationWithRAG(persona, company_context, vectorstore)
    
    print(f"\n{'='*50}")
    print(f"Persona: {persona['name']}")
    print(f"Difficulty: {persona['personality']['difficulty']}")
    print(f"RAG Enabled: {use_rag}")
    print(f"{'='*50}\n")
    print("YOU are the salesperson/service representative.")
    print("The AI will roleplay as the CUSTOMER.")
    print("The customer will initiate the conversation.\n")
    print("Type 'quit' to end the conversation\n")
    
    # Customer starts the conversation
    initial_prompt = f"You are starting a conversation with a {company_context['company_industry']} representative about {company_context['company_product']}. Begin the conversation as the customer would."
    first_message = conversation.send_message(initial_prompt)
    print(f"Customer: {first_message}\n")
    
    while True:
        user_input = input("You (Rep): ")
        
        # Check for quit BEFORE sending to AI
        if user_input.lower() == "quit":
            print("\nEnding conversation...")
            break
        
        # Only send non-quit messages
        if user_input.strip():  # Also ignore empty inputs
            response = conversation.send_message(user_input)
            print(f"\nCustomer: {response}\n")
    
    filename = conversation.save()
    print(f"\n{'='*50}")
    print("Conversation Complete!")
    print(f"Total exchanges: {conversation.get_exchange_count()}")
    print(f"Saved to: {filename}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_interactive_session_with_rag()