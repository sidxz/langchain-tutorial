from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()
# Documents can be used to store text and metadata together
# Can be filtered by metadata
doc = Document(
    page_content="This is my document. It is full of text that I've gathered from other places",
    metadata={
        "my_document_id": 234234,
        "my_document_source": "The LangChain Papers",
        "my_document_create_time": 1680013019,
    },
)

print(doc.page_content)