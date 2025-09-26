from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils import Config

chat = Config().new_openai_like()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English to French translator."),
        ("user", "Translate this to French: {text}"),
    ]
)

formatted_messages = template.format_messages(text="Hello, how are you?")
result = chat.invoke(formatted_messages)
print(result.content)
