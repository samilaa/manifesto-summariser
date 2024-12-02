import os
import sys

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import asyncio
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import EmbeddingResponse, Message, Role
from dotenv import load_dotenv
from pathlib import Path
from data.data_utils import from_pdf_to_string

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")

provider = OpenAIProvider(api_key=key)

# Files that are used to prompt the model
first_message_en = """
    Olet chatbotti, joka edustaa Suomen Keskustapuoluetta. Vastaa kohteliaasti äläkä 
    kiroilla tai muuten käyttää loukkaavaa kieltä. Puhu Suomea. Älä vastaa provokaatioihin 
    keskustelukumppanisi provokaatioyrityksiin. Vastaat kysymyksiin, jotka perustuvat arvoihin ja
    tavoitteita, jotka on esitetty tässä puoluemanifestissa:
"""

async def main():
    user_input = ""
    manifesto = from_pdf_to_string("data/manifestos/keskusta/keskusta-eduskuntavaaliohjelma-2023.pdf")
    prompt = first_message_en + manifesto

    # print(provider.estimated_cost(await provider.count_tokens(prompt), 500.0))
    # cost of sending the manifesto: 0.046865500000000004 (of ? currency)
    
    first_message = [Message(role=Role("system"), content=prompt)]

    response = await provider.generate(
            first_message,
            temperature=0.7,
            )

    print(response.content + "\n")

    while user_input != "Bye!":
        user_input = input("Kysy puolueelta: ")
        message = [Message(role=Role("user"), content=user_input)]
        response = await provider.generate(
                message,
                temperature=0.7,
                )
        print("\n" + response.content + "\n")

asyncio.run(main())
