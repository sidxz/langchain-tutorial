from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
# The vectorstore we'll be using
from langchain_community.vectorstores import FAISS
# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.callbacks.tracers import ConsoleCallbackHandler  # For debugging
from dotenv import load_dotenv

load_dotenv()


loader = WebBaseLoader("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10353056/")
docs = loader.load()

doc = docs[0].page_content

print (f"You have {len(docs)} document")
print (f"You have {len(doc)} characters in that document")

# Split your docs into texts
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=350, separators=["\n\n", "\n"])
texts = text_splitter.split_documents(docs)

# Get the total number of characters so we can see the average later
num_total_characters = sum([len(x.page_content) for x in texts])
print (f"Now you have {len(texts)} documents that have an average of {num_total_characters / len(texts):,.0f} characters (smaller pieces)")

# Embed your documents and combine with the raw text in a pseudo db (docsearch). 
# Note: This will make an API call to OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

# Create your retrieval engine
retriever = docsearch.as_retriever()
llm = ChatOpenAI(temperature=0, model="gpt-4o")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

question_answer_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

chain = create_retrieval_chain(retriever, question_answer_chain)

query = "What does the author say about data practices in academia ?"
res = chain.invoke({"input": query}, config={"callbacks": [ConsoleCallbackHandler()]})
print(res["answer"])
