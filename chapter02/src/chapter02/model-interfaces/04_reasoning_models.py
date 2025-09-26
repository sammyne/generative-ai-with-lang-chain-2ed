from langchain_core.prompts import ChatPromptTemplate
from utils import Config

# initialize OpenAI-like model with reasoning_effort parameter
# llm = Config().new_openai_like(reasoning_effort="high")
llm = Config().new_openai_like()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an experienced programmer and mathematical analyst."),
        ("user", "{problem}"),
    ]
)

# Initialize with reasoning_effort parameter
# chat = ChatOpenAI(
# model="o3-mini","
# reasoning_effort="high" # Options: "low", "medium", "high"
# )
chain = template | llm

problem = """
Design an algorithm to find the kth largest element in an unsorted array
with the optimal time complexity. Analyze the time and space complexity
of your solution and explain why it's optimal.
"""
response = chain.invoke({"problem": problem})
print(response.content)
