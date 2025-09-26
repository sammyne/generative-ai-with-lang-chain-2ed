from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from utils import Config
from operator import itemgetter

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

# Alternative approach using dictionary construction
manual_chain = (
    RunnablePassthrough()  # Pass through input
    | {
        "story": story_chain,  # Add story result
        "topic": itemgetter("topic"),  # Preserve original topic
    }
    | RunnablePassthrough().assign(  # Add analysis based on story
        analysis=analysis_chain
    )
)
result = manual_chain.invoke({"topic": "a rainy day"})
print(result.keys())  # Output: dict_keys(['story', 'topic', 'analysis'])
