from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Austin%E2%80%93Bergstrom_International_Airport")
docs = loader.load()

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

# Split your docs into texts
texts = text_splitter.split_documents(docs)

# Print the number of texts
print(f"Number of texts: {len(texts)}")

llm = ChatOpenAI(temperature=0, model="gpt-4o")
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

result = chain.invoke(docs)

print(result["output_text"])
