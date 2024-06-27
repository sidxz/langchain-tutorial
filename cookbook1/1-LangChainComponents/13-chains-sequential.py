# Chains ⛓️⛓️⛓️
# Combining different LLM calls and action automatically

# Ex: Summary #1, Summary #2, Summary #3 > Final Summary

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain.callbacks.tracers import ConsoleCallbackHandler  # For debugging

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=1, model="gpt-4o")
# First Prompt
location_template = """Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
location_prompt = PromptTemplate(
    input_variables=["user_location"], template=location_template
)

# Holds my 'location' chain
# location_chain = LLMChain(llm=llm, prompt=location_prompt)
location_chain = location_prompt | llm

# 2nd Prompt

meal_template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
meal_prompt = PromptTemplate(input_variables=["user_meal"], template=meal_template)

# Holds my 'meal' chain
meal_chain = meal_prompt | llm
# meal_chain = LLMChain(llm=llm, prompt=meal_prompt)

# overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
overall_chain = location_chain | meal_chain


review = overall_chain.invoke(
    {"user_location": "Cuttack"}, config={"callbacks": [ConsoleCallbackHandler()]}
)
print(review)
