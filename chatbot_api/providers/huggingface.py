from transformers import pipeline
from chatbot_api.base import LLMProvider

class HuggingFacerovider(LLMProvider):
    def __init__(self, model: str = "bigscience/bloom") -> None:
        self.generator = pipeline("text-generation", model=model)

    def generate_response(self, prompt: str) -> str:
        responses = self.generator(prompt, max_length=200, num_return_sequences=1)
        return responses[0]['generated_text']

    def get_usage_statistics(self) -> dict:
        return {"details": "Not implemented for Hugging Face backend"}
