# # Memory
# Helping LLMs remember information.
# Memory is a bit of a loose term. It could be as simple as remembering information
# you've chatted about in the past or more complicated information retrieval.

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

chat = ChatOpenAI(model="gpt-4o")

history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of france?")

print("---Initial---")
print(history)

chat_response = chat.invoke(history.messages)
print("---Chat Response---")
print(chat_response)

history.add_ai_message(chat_response.content)
print("---Updated---")
print(history.messages)
