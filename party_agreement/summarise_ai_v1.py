import pandas as pd
import os
import sys
# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import EmbeddingResponse, Message, Role
from dotenv import load_dotenv
from pathlib import Path
import asyncio

party = 'Vihreät'

async def main():
    # Set up the environment and file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'outputs', f'filtered_output_{party}.csv')
    
    # Read the CSV file
    # Note: Ensure encoding and separator if needed.
    df = pd.read_csv(csv_path)
    
    # We'll construct a large prompt string that includes all the data.
    # Each row corresponds to a candidate/entry. We'll present them as such:
    
    text_block = ["""Prompt:
You are given raw data containing the answers of all candidates from a political party to various policy questions. Each answer is on a scale of 1 to 5, where:
1 = Strongly Disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly Agree

Group the responses into key topics, such as:

Social Issues (e.g., gender policies, public services, equality)
Environmental and Climate Policies
Economic Issues (e.g., welfare, taxation, subsidies)
Immigration and Foreign Policy
For each topic:

Calculate the average response for all candidates from the party.
Assign a general stance:
Strongly Support for averages > 4
Support for averages > 3 and ≤ 4
Neutral for averages > 2 and ≤ 3
Oppose for averages > 1 and ≤ 2
Strongly Oppose for averages ≤ 1
Based on the assigned stance, produce a concise and factual summary that captures the party's ideological position and priorities for each topic. Ensure the summary is neutral and highlights overarching patterns, without listing numerical values.
Output Example:
"[Party Name] holds a neutral position on social issues, indicating mixed or balanced views on topics like gender policies and public services. They support environmental and climate policies, reflecting a general alignment with measures to address environmental concerns, though not with strong urgency. On economic issues, their stance is neutral, suggesting they take a pragmatic or case-by-case approach to welfare and economic subsidies. Lastly, they are strongly supportive of immigration and foreign policy measures, advocating for policies that likely emphasize openness to skilled immigration and stricter handling of law enforcement-related issues for immigrants."

Task: Create a similar summary for the party based on the raw data provided. The format should follow EXACTLY the above output example format.\n"""]
    
    # Identify the question columns (all except 'vaalipiiri', 'puolue')
    question_columns = [col for col in df.columns if col not in ['vaalipiiri', 'puolue']]
    
    for i, row in df.iterrows():
        vaalipiiri = row['vaalipiiri']
        puolue = row['puolue']
        
        text_block.append(f"Candidate from {vaalipiiri}, Party: {puolue}")
        for q_col in question_columns:
            question = q_col  # The header itself is the question/statement.
            answer = row[q_col]
            text_block.append(f"Q: {question}")
            text_block.append(f"A: {answer}")
            text_block.append("")  # Blank line for readability
    
    prompt_text = "\n".join(text_block)
    
    # Initialize the OpenAIProvider
    key = os.getenv("OPENAI_API_KEY")

    provider = OpenAIProvider(api_key=key)

    # Create the message list for the LLM
    messages = [
        Message(role=Role.USER, content=prompt_text)
    ]

    # Call the provider
    response = await provider.generate(messages=messages, temperature=0.7, max_tokens=500)

    # Print the LLM response
    print("LLM Summary:\n", response.content)

# Run the async main
if __name__ == "__main__":
    asyncio.run(main())