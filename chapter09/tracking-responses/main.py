"""Tracing of agent calls and intermediate results."""

import subprocess
from urllib.parse import urlparse

# from langchain.agents import AgentType, initialize_agent
# from langchain_core.tools import StructuredTool
from pydantic import HttpUrl
from utils import Config
from langgraph.prebuilt import create_react_agent


def ping(url: HttpUrl, return_error: bool) -> str:
    """Ping the fully specified url. Must include https:// in the url."""
    hostname = urlparse(str(url)).netloc
    completed_process = subprocess.run(
        ["ping", "-c", "1", hostname], capture_output=True, text=True
    )
    output = completed_process.stdout
    if return_error and completed_process.returncode != 0:
        return completed_process.stderr
    return output


llm = Config().new_openai_like()

agent = create_react_agent(
    model=llm, tools=[ping], prompt="You are a helpful assistant"
)

# 参考 https://python.langchain.com/docs/how_to/migrate_agent/#return_intermediate_steps
# 输出的 result 已包含所有中间执行步骤
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the latency like for https://langchain.com?",
            }
        ]
    }
)
print(result)
