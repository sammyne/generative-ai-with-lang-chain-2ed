# LiteLLM with LangChain
import os
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router
from langchain_core.prompts import PromptTemplate
import dotenv

dotenv.load_dotenv()

# Configure multiple model deployments with fallbacks
# openai/ 前缀的必要性参见 https://docs.litellm.ai/docs/providers/openai_compatible
model_list = [
    {
        "model_name": f"anthropic/{os.environ['ANTHROPIC_MODEL']}",
        "litellm_params": {
            "model": f"anthropic/{os.environ['ANTHROPIC_MODEL_FALLBACK']}",  # Automatic fallback option
            "api_key": os.environ["ANTHROPIC_API_KEY"],
            "api_base": os.environ["ANTHROPIC_BASE_URL"],
        },
    },
    {
        "model_name": f"openai/{os.environ['OPENAI_MODEL']}",
        "litellm_params": {
            "model": f"openai/{os.environ['OPENAI_MODEL_FALLBACK']}",  # Automatic fallback option
            "api_key": os.environ["OPENAI_API_KEY"],
            "api_base": os.environ["OPENAI_API_BASE_URL"],
        },
    },
]

# Setup router with reliability features
router = Router(
    model_list=model_list,
    routing_strategy="usage-based-routing-v2",
    cache_responses=True,  # Enable caching
    num_retries=3,  # Auto-retry failed requests
)

model_name = f"openai/{os.environ['OPENAI_MODEL']}"
# Create LangChain LLM with router
router_llm = ChatLiteLLMRouter(router=router, model_name=model_name)

# Build and use a LangChain
prompt = PromptTemplate.from_template("Summarize: {text}")
chain = prompt | router_llm
result = chain.invoke({"text": "LiteLLM provides reliability for LLM applications"})
print(result)
