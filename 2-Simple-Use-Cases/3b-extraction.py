# Extraction is the process of parsing data from a piece of 
# text. This is commonly used with output parsing in order to structure our data.
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
# How you would like your response structured. 
# This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="artist", description="The name of the musical artist"),
    ResponseSchema(name="song", description="The name of the song that the artist plays")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print (format_instructions)


prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("Given a command from the user, extract the artist and song names \n \
                                                    {format_instructions}\n{user_prompt}")  
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

song_name = """
Kana Yaari
Song by Eva B, Kaifi Khalil, and Wahab Bugti
"""

final_prompt = prompt.format(user_prompt=song_name)

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
