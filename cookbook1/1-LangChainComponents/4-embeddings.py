from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

# Text Embedding Model
# Change your text into a vector (a series of numbers that hold the semantic 'meaning'
# of your text). Mainly used when comparing two pieces of text together.


embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

text = "Hi, how are you?"
text_embedding = embeddings.embed_query(text)

print(text_embedding)
print(f"Your embedding length is: {len(text_embedding)}")
