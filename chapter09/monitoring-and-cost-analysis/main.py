from langchain_community.callbacks.manager import get_openai_callback
from utils import Config

# llm = ChatOpenAI(model="gpt-4o")
llm = Config().new_openai_like()

with get_openai_callback() as cb:
    response = llm.invoke("Explain quantum computing in simple terms")

    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")