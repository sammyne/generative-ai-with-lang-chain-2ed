import os

import dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class Config:
    def __init__(self):
        # By default, load_dotenv doesn't override existing environment variables and looks for a .env file in same directory as python script or searches for it incrementally higher up.
        dotenv_path = dotenv.find_dotenv(usecwd=True)
        if not dotenv_path:
            raise ValueError("No .env file found")
        dotenv.load_dotenv(dotenv_path=dotenv_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        base_url = os.getenv("OPENAI_API_BASE_URL")
        if not base_url:
            raise ValueError("OPENAI_API_BASE_URL is not set")

        model = os.getenv("OPENAI_MODEL")
        if not model:
            raise ValueError("OPENAI_MODEL is not set")

        embeddings_model = os.getenv("OPENAI_EMBEDDINGS_MODEL")
        hf_pretrained_embeddings_model = os.getenv("HF_PRETRAINED_EMBEDDINGS_MODEL")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.embeddings_model = embeddings_model
        self.hf_pretrained_embeddings_model = (
            hf_pretrained_embeddings_model
            if hf_pretrained_embeddings_model
            else "Qwen/Qwen3-Embedding-8B"
        )

    def new_openai_like(self, **kwargs) -> ChatOpenAI:
        # 参考：https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2587654
        # 参考：https://help.aliyun.com/zh/model-studio/models
        # ChatOpenAI 文档参考：https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI
        return ChatOpenAI(
            api_key=self.api_key, base_url=self.base_url, model=self.model, **kwargs
        )

    def new_openai_like_embeddings(self, **kwargs) -> OpenAIEmbeddings:
        if not self.embeddings_model:
            raise ValueError("OPENAI_EMBEDDINGS_MODEL is not set")

        # 参考：https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html#langchain_openai.embeddings.base.OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.embeddings_model,
            # https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html#langchain_openai.embeddings.base.OpenAIEmbeddings.tiktoken_enabled
            # 对于非 OpenAI 的官方实现，将这个参数置为 False。
            # 回退到用 huggingface transformers 库 AutoTokenizer 来处理 token。
            tiktoken_enabled=False,
            # https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html#langchain_openai.embeddings.base.OpenAIEmbeddings.model
            # 元宝说 Jina 的 embedding 模型 https://huggingface.co/jinaai/jina-embeddings-v4 最接近
            # text-embedding-ada-002
            # 个人喜好，选了 Qwen/Qwen3-Embedding-8B
            # tiktoken_model_name='Qwen/Qwen3-Embedding-8B',
            tiktoken_model_name=self.hf_pretrained_embeddings_model,
            **kwargs,
        )
