from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_chain

from dotenv import load_dotenv
load_dotenv()


# Let's get started by defining a simple model for us to extract from.
# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field



# Set up a parser + inject instructions into the prompt template.
output_parser = PydanticOutputParser(pydantic_object=Joke)
format_instructions = output_parser.get_format_instructions()

print("-----format_instructions-----")
print (format_instructions)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)


final_prompt = prompt.format(query="Tell me a joke about sandwich.")
print("-----final_prompt-----")
print(final_prompt)

llm = ChatOpenAI(temperature=0.5)

chain = prompt | llm | output_parser
res = chain.invoke(final_prompt)
print("-----res------")
print(res)