#  LangChain Components
#  Schema - Nuts and Bolts of working with Large Language Models (LLMs)
#  Text

# Chat Messages
# Like text, but specified with a message type (System, Human, AI)

# System - Helpful background context that tell the AI what to do
# Human - Messages that are intented to represent the user
# AI - Messages that show what the AI responded with

from dotenv import load_dotenv

load_dotenv()


from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# This it the language model we'll use. We'll talk about what we're doing below in the next section
llm = ChatOpenAI(temperature=0.7)

query1 = [
    SystemMessage(
        content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"
    ),
    HumanMessage(content="I like tomatoes, what should I eat?"),
]
res = llm.invoke(query1)
print(res.content)


query2 = [
    SystemMessage(
        content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"
    ),
    HumanMessage(content="I like the beaches where should I go?"),
    AIMessage(content="You should go to Nice, France"),
    HumanMessage(content="What else should I do when I'm there?"),
]

res = llm.invoke(query2)
print(res.content)
