"""Run this model in Python

> pip install openai
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings. 
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)

response = client.responses.create(
    model="gpt-4o-mini",
    input="tell me a joke",
)

print(response.output[0].content[0].text)
# Why did the scarecrow win an award?
# Because he was outstanding in his field!


# Fetch the response using the response ID - the response api is statefull
fetched_response = client.responses.retrieve(
response_id=response.id)

print(fetched_response.output[0].content[0].text)
# Why did the scarecrow win an award?
# Because he was outstanding in his field!

# You can now also reference the response from previous conversation

response_two = client.responses.create(
    model="gpt-4o-mini",
    input="tell me another",
    previous_response_id=response.id
)

print(response_two.output[0].content[0].text)

# Why don't skeletons fight each other?
# They don't have the guts!


response = client.responses.create(
    model="gpt-4o",  # or another supported model
    input="What's the latest news about AI?",
    tools=[
        {
            "type": "web_search"
        }
    ]
)

import json
print(json.dumps(response.output, default=lambda o: o.__dict__, indent=2))