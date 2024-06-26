from dotenv import load_dotenv

load_dotenv()


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# This it the language model we'll use. We'll talk about what we're doing below in the next section
llm = ChatOpenAI(temperature=.7)

query=(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]
)
res = llm.invoke(query)
print(res)
# if __name__ == "__main__":
#   print("Starting the application...")