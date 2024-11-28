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
from pypdf import PdfReader


# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")

provider = OpenAIProvider(api_key=key)

# Files that are used to prompt the model
first_message_en = """
    You are a chatbot representing the Centre party from Finland. Answer politely and do not 
    swear or otherwise use offensive language. Speak English. Do not respond to provocation 
    attemps by your conversation partner. You will answer questions based on the values and
    goals outlined in this manifesto:
"""

async def main():
    user_input = ""
    # TODO: look into cleaning the input text
    reader = PdfReader(Path(__file__).resolve().parents[1] / "data/manifestos/keskusta/keskusta-eduskuntavaaliohjelma-2023.pdf")
    manifesto = ""
    for page in reader.pages:
        manifesto += page.extract_text()
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
        user_input = input("Ask the party: ")
        message = [Message(role=Role("user"), content=user_input)]
        response = await provider.generate(
                message,
                temperature=0.7,
                )
        print("\n" + response.content + "\n")

asyncio.run(main())
