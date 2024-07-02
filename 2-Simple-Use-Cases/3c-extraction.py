# Extraction is the process of parsing data from a piece of 
# text. This is commonly used with output parsing in order to structure our data.

# Doing the same thing as 1 but a hack for parsing List of JSON objects

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
# How you would like your response structured. 
# This is basically a fancy prompt template
response_schemas_json = [
    ResponseSchema(name="fruitName", description="This is the input fruit name"),
    ResponseSchema(name="fruitEmoji", description="This is your response, a emoji for the fruit")
]
res_json = """
response_schemas_json
```json
{
        "fruitName": string  // This is the input fruit name
        "fruitEmoji": string  // This is your response, a emoji for the fruit
}
```"""
response_schemas = [
    ResponseSchema(name="result", description="This is an array of json objects having fruit names and emojis:" + res_json, type="List[response_schemas_json]"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print (format_instructions)



prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("You will be given a sentence with fruit names, extract those fruit names and assign an emoji to them \n \
                                                    {format_instructions}\n{user_prompt}")  
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

fruit_names = """
Apple, Pear, this is an kiwi
"""

final_prompt = prompt.format(user_prompt=fruit_names)

print("==== PROMPT SENT TO LLM ====")
print(final_prompt)


llm_res = llm.invoke(final_prompt)

print("==== ORIGINAL LLM RESPONSE ====")
print(type(llm_res.content))
print(llm_res.content)



#Now parse the output
parsed_res = output_parser.parse(llm_res.content)
print("==== PARSED RESPONSE ====")
print(type(parsed_res))
print(parsed_res)
