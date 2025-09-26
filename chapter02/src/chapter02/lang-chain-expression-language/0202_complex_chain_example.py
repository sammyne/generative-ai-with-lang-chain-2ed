from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from utils import Config

chat = Config().new_openai_like()

# First chain generates a story
story_prompt = PromptTemplate.from_template("Write a short story about {topic}")
story_chain = story_prompt | chat | StrOutputParser()

# Second chain analyzes the story
analysis_prompt = PromptTemplate.from_template(
    "Analyze the following story's mood:\n{story}"
)
analysis_chain = analysis_prompt | chat | StrOutputParser()

output_prompt = PromptTemplate.from_template(
    "Here's the story: \n{story}\n\nHere's the mood: \n{mood}"
)

# Using RunnablePassthrough.assign to preserve data
enhanced_chain = RunnablePassthrough.assign(
    story=story_chain  # Add 'story' key with generated content
).assign(
    analysis=analysis_chain  # Add 'analysis' key with analysis of the story
)

result = enhanced_chain.invoke({"topic": "a rainy day"})
print(result.keys())  # Output: dict_keys(['topic', 'story', 'analysis'])
# dict_keys(['topic', 'story', 'analysis'])
