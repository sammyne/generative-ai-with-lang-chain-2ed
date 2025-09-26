from langchain_core.messages import HumanMessage, SystemMessage
from utils import Config

# initialize OpenAI-like model
llm = Config().new_openai_like()

messages = [
    SystemMessage(content="You're a helpful programming assistant"),
    HumanMessage(content="Write a Python function to calculate factorial"),
]
response = llm.invoke(messages)
print(response.content)
