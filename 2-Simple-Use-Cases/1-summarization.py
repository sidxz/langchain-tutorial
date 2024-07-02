# Recap of 14-chain-summary

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

loader = WebBaseLoader("https://saclab.github.io/daikon/docs/intro")
docs = loader.load()
full_text = docs[0].page_content

llm = ChatOpenAI(temperature=0, model="gpt-4o")

num_of_tokens = llm.get_num_tokens(full_text)
print(f"Number of tokens: {num_of_tokens}")

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=350, separators=["\n\n", "\n"])

# Split your docs into texts
texts = text_splitter.split_documents(docs)

# Print the number of texts
print(f"Number of texts: {len(texts)}")


chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

result = chain.invoke(docs)

print(result["output_text"])
