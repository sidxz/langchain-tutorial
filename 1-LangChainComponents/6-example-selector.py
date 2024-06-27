# Example Selectors
# An easy way to select from a series of examples that allow you to dynamic
# place in-context information into your prompt. Often used when your task
# is nuanced or you have a large list of examples.

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Example Input: {input}\nExample Output: {output}",
)

# Examples of locations that nouns are found
examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]

# SemanticSimilarityExampleSelector will select examples that are similar to your
# input by semantic meaning

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are 
    # used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings
    # and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=2,
)

similar_prompt = FewShotPromptTemplate(
    # The object that will help select examples
    example_selector=example_selector,
    
    # Your prompt
    example_prompt=example_prompt,
    
    # Customizations that will be added to the top and bottom of your prompt
    prefix="Give the location an item is usually found in",
    suffix="Input: {noun}\nOutput:",
    
    # What inputs your prompt will receive
    input_variables=["noun"],
)


# Select a noun!
my_noun = "captain"
# my_noun = "student"

# This is where it is executed, based on the noun you provided,
# the examples are selected and the prompt is generated.
final_prompt = similar_prompt.format(noun=my_noun)
print(final_prompt)

llm = ChatOpenAI()
res = llm.invoke(final_prompt)

print("-----------")
print(res.content)
