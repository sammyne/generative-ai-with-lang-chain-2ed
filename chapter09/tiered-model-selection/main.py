from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import os

dotenv.load_dotenv()

base_url = os.environ["OPENAI_API_BASE_URL"]

# Define models with different capabilities and costs
affordable_model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL_AFFORDABLE"],
    base_url=base_url,
)  # ~10× cheaper than gpt-4o

powerful_model = ChatOpenAI(
    model=os.environ["OPENAI_MODEL_POWERFUL"],
    base_url=base_url,
)  # More capable but more expensive

# Create classifier prompt
classifier_prompt = ChatPromptTemplate.from_template(
    """
Determine if the following query is simple or complex based on these
criteria:
- Simple: factual questions, straightforward tasks, general knowledge
- Complex: multi-step reasoning, nuanced analysis, specialized expertise

Query: {query}

Respond with only one word: "simple" or "complex"
"""
)

# Create the classifier chain
classifier = classifier_prompt | affordable_model | StrOutputParser()


def route_query(query):
    """Route the query to the appropriate model based on complexity."""
    complexity = classifier.invoke({"query": query})

    if "simple" in complexity.lower():
        print(f"Using affordable model for: {query}")
        return affordable_model
    else:
        print(f"Using powerful model for: {query}")
        return powerful_model


# Example usage
def process_query(query):
    model = route_query(query)
    return model.invoke(query)


simple_query = "what is the sum of 2+3"
print(process_query(simple_query))

# complex_query = "plan a app serving billion users"
# print(process_query(complex_query))