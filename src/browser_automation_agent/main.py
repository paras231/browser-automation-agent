from ollama import Client
import os
from dotenv import load_dotenv
import requests
import ollama
from tools import open_page,type_text,click_element,observe_page,take_screenshot

load_dotenv()

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

print(os.environ.get("OLLAMA_API_KEY"))
client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY',"")}
)

messages = [
  {
    'role': 'user',
    'content': 'open amazon.com and search for men shoes',
  },
]

#  tool calling 
# 
# this tool uses Ollama web search API
def get_country_info(country: str):
    """Returns information about a country."""

    query = f"Tell me about {country}"

    response = requests.post(
        "https://ollama.com/api/web_search",
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "query": query
        }
    )

    response.raise_for_status()

    return response.json()


tools = [
    open_page,
    observe_page,
    type_text,
    click_element,
    take_screenshot
]



# print(response)

tool_map = {
    "open_page": open_page,
    "observe_page":observe_page,
    "click_element": click_element,
    "type_text": type_text,
    "take_screenshot":take_screenshot
}

while True:

    response = client.chat(
        "gemma4:31b",
        messages=messages,
        tools=tools
    )

    messages.append(response.message)

    # Model has finished
    if not response.message.tool_calls:
        print("FINAL:")
        print(response.message.content)
        break

    for tool_call in response.message.tool_calls:

        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        print("\nCalling:", function_name)
        print("Arguments:", arguments)

        tool = tool_map[function_name]

        result = tool(**arguments)

        print("Result:", result)

        # VERY IMPORTANT
        messages.append({
            "role": "tool",
            "name": function_name,
            "content": str(result)
        })