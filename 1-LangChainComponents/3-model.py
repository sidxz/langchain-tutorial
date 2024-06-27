# Models - The interface to the AI brains
# Language Model
# A model that does text in ➡️ text out!

# Check out how I changed the model I was using 
# from the default one to gpt-4o

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
query1 = ('What comes after Friday?')

res = llm.invoke(query1)
print(res.content)

query2 =(
    [
        SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
        HumanMessage(content="I would like to go to New York, how should I do this?")
    ]
)

res = llm.invoke(query2)
print(res.content)
