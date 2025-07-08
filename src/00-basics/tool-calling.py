import os
import json
from openai import OpenAI
import pytz
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

# Define a function that returns flight information between two cities (mock implementation)
def get_current_time(city_name: str) -> str:
    """Returns the current time in a given city."""
    try:
        print("get current time for location: ", city_name)
        # Get the timezone for the city
        timezone = pytz.timezone(city_name)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except Exception as e:
        print("Error: ", e)
        return "Sorry, I couldn't find the time for that city."

# Define a function tool that the model can ask to invoke in order to retrieve flight information
tool={
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": """Returns information about the current time in a specific cities. The city format should be like Europe/Berlin for every city that is placed in Germany. Other possible format are Europe/Paris, America/New_York, Asia/Bangkok, America/Seattle""",
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "The name of the city",
                }
            },
            "required": [
                "city_name",
            ],
        },
    },
}

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

messages=[
    {"role": "system", "content": "You are an assistant that helps users find information about places. The user is in Cologne"},
    {"role": "user", "content": "What is currently morning or afternoon?"},
]

response = client.chat.completions.create(
    messages=messages,
    tools=[tool],
    model=model_name,
)

# We expect the model to ask for a tool call
if response.choices[0].finish_reason == "tool_calls":

    # Append the model response to the chat history
    messages.append(response.choices[0].message)

    # We expect a single tool call
    if response.choices[0].message.tool_calls and len(response.choices[0].message.tool_calls) == 1:

        tool_call = response.choices[0].message.tool_calls[0]

        # We expect the tool to be a function call
        if tool_call.type == "function":

            # Parse the function call arguments and call the function
            function_args = json.loads(tool_call.function.arguments.replace("'", '"'))
            print(f"Calling function `{tool_call.function.name}` with arguments {function_args}")
            callable_func = locals()[tool_call.function.name]
            function_return = callable_func(**function_args)
            print(f"Function returned = {function_return}")

            # Append the function call result fo the chat history
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": function_return,
                }
            )

            # Get another response from the model
            response = client.chat.completions.create(
                messages=messages,
                tools=[tool],
                model=model_name,
            )

            print(f"Model response = {response.choices[0].message.content}")