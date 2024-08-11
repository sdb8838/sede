import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv


load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
"""
model = "mistral-large-latest"


client = MistralClient(api_key=api_key)

chat_response = client.chat(
    model=model,
    messages=[ChatMessage(role="user", content="What is the best French cheese?")]
)

print(chat_response.choices[0].message.content)

"""

model = "mistral-embed"

client = MistralClient(api_key=api_key)

embeddings_response = client.embeddings(
    model=model,
    input=["Embed this sentence.", "As well as this one."]
)

print(embeddings_response)