from utils import Config

c = Config()

# initialize OpenAI-like model
llm = c.new_openai_like()

# Both can be used with the same interface
response = llm.invoke("Tell me a joke about light bulbs!")
print(response.content)
