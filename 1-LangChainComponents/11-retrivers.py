# Text splitters are used to split text into smaller parts.
from langchain_community.document_loaders import GutenbergLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/1522/pg1522.txt")

data = loader.load()
# print the datatype
full_text = data[0].page_content

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1500,
    chunk_overlap  = 20,
)

documents = text_splitter.create_documents([full_text])

# get the embeddings
embeddings = OpenAIEmbeddings()

# db with embeddings
db = FAISS.from_documents(documents, embeddings)

# Init your retriever
retriever = db.as_retriever()

query = "Who was the emperor of Rome?"
docs = retriever.invoke(query, k=1) # k=1 means we only want the top result
print(docs[0].page_content)
