import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Define an analytical tool that the model can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_row_count",
            "description": "Get the total number of rows in an uploaded dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the uploaded dataset."
                    }
                },
                "required": ["dataset_name"]
            }
        }
    }
]

messages = [
    {
        "role": "system",
        "content": (
            "You are DataSense AI, an analytics assistant. "
            "When asked for the row count of a dataset, "
            "you must call the get_dataset_row_count tool."
        )
    },
    {
        "role": "user",
        "content": "How many rows are in sample_sales.csv?"
    }
]

response = client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=messages,
    tools=tools,
    tool_choice="required",
    max_tokens=200
)

message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        print("Tool selected:", tool_call.function.name)
        print("Arguments:", tool_call.function.arguments)
else:
    print("Model response:", message.content)