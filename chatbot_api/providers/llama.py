import requests
from chatbot_api.base import LLMProvider

class LlamaBackend(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.llama.ai/v1"  # Hypothetical URL

    def generate_response(self, prompt: str) -> str:
        """Send the prompt to LLaMA API and get the response."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        response = requests.post(f"{self.base_url}/generate", json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"LLaMA API request failed: {response.status_code} - {response.text}")
        return response.json()["text"]
