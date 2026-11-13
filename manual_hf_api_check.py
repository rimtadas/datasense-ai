import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load credentials from .env
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Check your .env file.")

# Connect to Hugging Face
client = InferenceClient(token=hf_token)

# Send a test message to an LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=[
        {
            "role": "user",
            "content": "Explain what a data analyst does in two sentences."
        }
    ],
    max_tokens=150,
)

print(response.choices[0].message.content)