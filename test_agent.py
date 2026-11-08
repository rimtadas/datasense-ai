import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Actual Python analytical function
def get_dataset_row_count(dataset_name):
    if dataset_name != "sample_sales.csv":
        raise ValueError("Dataset not found.")

    file_path = Path(__file__).parent / dataset_name
    df = pd.read_csv(file_path)

    return {"dataset_name": dataset_name, "row_count": len(df)}


# Define the tool for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_row_count",
            "description": "Get the total number of rows in an uploaded dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"}
                },
                "required": ["dataset_name"]
            }
        }
    }
]

messages = [
    {
        "role": "system",
        "content": "You are DataSense AI. Use the available analytical tools when needed."
    },
    {
        "role": "user",
        "content": "How many rows are in sample_sales.csv?"
    }
]

# Step 1: Ask the model to decide which tool to use
response = client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=messages,
    tools=tools,
    tool_choice="required",
    max_tokens=200
)

message = response.choices[0].message

# Step 2: Execute the selected Python function
for tool_call in message.tool_calls:
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "get_dataset_row_count":
        result = get_dataset_row_count(**arguments)
    else:
        raise ValueError(f"Unknown tool: {function_name}")

    print("Python tool executed:", function_name)
    print("Tool result:", result)

    # Step 3: Send the result back to the model
    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": tool_call.function.arguments
                }
            }
        ]
    })

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

# Step 4: Let the model formulate the final answer
final_response = client.chat.completions.create(
    model="openai/gpt-oss-120b:fastest",
    messages=messages,
    tools=tools,
    max_tokens=200
)

print("\nDataSense AI:", final_response.choices[0].message.content)