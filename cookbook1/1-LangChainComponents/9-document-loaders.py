# Data loaders
from langchain_community.document_loaders import HNLoader

from dotenv import load_dotenv
load_dotenv()

loader=HNLoader("https://news.ycombinator.com/item?id=40802423")

data = loader.load()

print("-----data------")
#print(data)
print(f"Found {len(data)} items.")