from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import Config

# Initialize the LLM with max_tokens parameter
# llm = ChatOpenAI(model="gpt-4o", max_tokens=150)  # Limit to approximately 100-120 words
llm = Config().new_openai_like(max_tokens=150)

# Create a prompt template with length guidance
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that provides concise, accurate information. Your responses should be no more than 100 words unless explicitly asked for more detail.",
        ),
        ("human", "{query}"),
    ]
)

# Create a chain
chain = prompt | llm | StrOutputParser()

result = chain.invoke(
    {"query": "write simple python function checking is a integer is prime"}
)
print(result)
