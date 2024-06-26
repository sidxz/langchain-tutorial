from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
# Prompt Template
# An object that helps create prompts based on a combination of user input,
# other non-static information and a fixed template string.

template = """
I really want to travel to {location}. What should I do there?

Respond in one short sentence
"""

prompt = PromptTemplate(
    input_variables=["location"],
    template=template,
)

final_prompt = prompt.format(location="Bhubaneswar")
print(f"Final Prompt: {final_prompt}")

llm = ChatOpenAI()
res = llm.invoke(final_prompt)

print("-----------")
print(res.content)
