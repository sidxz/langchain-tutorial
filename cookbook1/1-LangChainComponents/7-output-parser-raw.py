from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# How you would like your response structured. 
# This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
print (format_instructions)



template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

final_prompt = prompt.format(user_input="welcom to texsa!")
print("-----final_prompt-----")
print(final_prompt)

llm = ChatOpenAI()
llm_res = llm.invoke(final_prompt)

print("-----res------")
print(llm_res.content)

parsed_res = output_parser.parse(llm_res.content)
print("-----parsed_res------")
print(parsed_res)