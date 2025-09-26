from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
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
# Combine chains
story_with_analysis = story_chain | analysis_chain

# Run the combined chain
result = story_with_analysis.invoke({"topic": "a rainy day"})
print(result)
