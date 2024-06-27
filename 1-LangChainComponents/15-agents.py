# Some applications will require not just a predetermined chain of calls to LLMs/other tools,
# but potentially an unknown chain that depends on the user's input. In these types of chains,
# there is a “agent” which has access to a suite of tools. Depending on the user input,
# the agent can then decide which, if any, of these tools to call.

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

import json

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)

# pipenv install google-search-results
tools = load_tools(["serpapi"], llm=llm)

# prompt
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

res = agent_executor.invoke(
    {
        "input": "what is the capital of the state where the longest running Chief Minister of Orissa was born? Make sure your answer is a valid capital city of India."
    }
)

print(res)
