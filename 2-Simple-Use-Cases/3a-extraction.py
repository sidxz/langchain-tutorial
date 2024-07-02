# Extraction is the process of parsing data from a piece of 
# text. This is commonly used with output parsing in order to structure our data.
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

instructions = """
You will be given a sentence with fruit names, extract those fruit names and assign an emoji to them
Return the fruit name and emojis in a python dictionary
"""

fruit_names = """
Apple, Pear, this is an kiwi
"""

# Make your prompt which combines the instructions w/ the fruit names
prompt = (instructions + fruit_names)

llm_res = llm.invoke(prompt)

print("-----res------")
print(llm_res.content)

print("-----type------")
print(type(llm_res.content))



# How you would like your response structured. 
# This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="fruitName", description="This is the input fruit name"),
    ResponseSchema(name="fruitEmoji", description="This is your response, a emoji for the fruit")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
print (format_instructions)