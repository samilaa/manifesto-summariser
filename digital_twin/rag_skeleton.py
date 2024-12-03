import os
import sys

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import asyncio
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import Message, Role, LLMResponse
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from typing import List, Optional
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")

@dataclass
class PartyContext:
    """Stores the context information for a political party"""
    party_name: str
    context_text: str

class PoliticalChatbot:
    def __init__(self, openai_provider: OpenAIProvider):
        """
        Initialize the political chatbot with an OpenAI provider
        
        Args:
            openai_provider: Instance of OpenAIProvider for making API calls
        """
        self.llm = openai_provider
        self.party_contexts: Dict[str, PartyContext] = {}
        
    def add_party_context(self, text: str, party_name: str) -> None:
        """
        Add or update context for a political party
        
        Args:
            text: The unstructured text data about the party
            party_name: Name of the political party
        """
        self.party_contexts[party_name.lower()] = PartyContext(
            party_name=party_name,
            context_text=text
        )

    def _detect_party_in_query(self, query: str) -> Optional[str]:
        """
        Detect which party is being discussed in the user's query
        
        Args:
            query: The user's question or message
            
        Returns:
            The detected party name or None if no match
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check if any party name appears in the query
        for party_name in self.party_contexts.keys():
            if re.search(rf'\b{party_name}\b', query_lower):
                return party_name
        return None

    async def get_response(self, query: str, chat_history: List[Message]) -> LLMResponse:
        """
        Generate a response to the user's query about a political party
        
        Args:
            query: The user's question or message
            chat_history: List of previous messages in the conversation
            
        Returns:
            The chatbot's response
        """
        party_name = self._detect_party_in_query(query)
        
        if not party_name or party_name not in self.party_contexts:
            return "Puolue ei ole tiedossani tai se on kirjoitettu muuten kuin perusmuodossa. Voisitko tarkentaa kysymystäsi jonkin tunnetun puolueen nimellä?"

        # Construct the system message with party context
        system_message = f"""Olet {self.party_contexts[party_name].party_name}:n poliittisten kantojen ja arvojen asiantuntija. 
            Perusta vastauksesi seuraavaan asiayhteyteen ja säilytä samalla neutraali ja analyyttinen sävy:
            
            {self.party_contexts[party_name].context_text}
            
            Anna asiallisia, hyvin perusteltuja vastauksia, jotka perustuvat yksinomaan annettuun kontekstiin. Jos kysymys ylittää käytettävissä olevat tiedot,
            myönnä annetun asiayhteyden rajoitukset."""
        
        # Combine system message, chat history, and current query
        messages = [Message(Role("system"), system_message)] + chat_history + [Message(Role("user"), query)] # chat history is a list of Messages
        
        try:
            response = await self.llm.generate(
                messages, 
                temperature=0.7  # Use moderate temperature for balanced responses
            )
            return response
        except Exception as e:
            return f"Virhe: {str(e)}"

# Example usage
async def main():
    # Initialize the chatbot
    openai_provider = OpenAIProvider(api_key=key)
    chatbot = PoliticalChatbot(openai_provider)

    # Initialize chat
    system_prompt = """
    Olet chatbotti, jonka tarkoituksena on auttaa käyttäjää ymmärtämään eri puolueiden tulevaisuuden 
    suunnitelmia sekä poliittisia kantoja ja arvoja. Puhu Suomea. Älä vastaa provokaatioihin. 
    """ # TODO: improve system prompt

    chat_history = [Message(Role("system"), system_prompt)] # initialize with system prompt
    user_input = ""

    response = await openai_provider.generate(
            chat_history,
            temperature=0.7,
            )

    print(f'Olen chatbotti, joka auttaa sinua ymmärtämään eri puolueita. Mistä puolueesta haluaisit tietää lisää?')

    # Add context for different parties
    chatbot.add_party_context(
        text="Kokoomus uskoo verojen nostamiseen köyhille ja verojen laskemiseen rikkaille.", # the manifestos (maybe do this in a separate file)
        party_name="Kokoomus" # party name
    )
    
    while user_input != "Bye!":
        # Get response to a query
        query = input("Käyttäjä:")
        response = await chatbot.get_response(query, chat_history)
        print("\n" + response.content + "\n")

        # Update chat history
        chat_history.extend([
            [Message(Role("user"), query)],
            [Message(Role("user"), response.content)]
        ])

if __name__ == "__main__":
    asyncio.run(main())