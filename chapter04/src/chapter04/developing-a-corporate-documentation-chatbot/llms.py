"""Loading LLMs and Embeddings."""

from config import Config
import config
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

chat_model = Config().new_openai_like(
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

store = LocalFileStore("./cache/")

# underlying_embeddings = Config().new_openai_like_embeddings()
underlying_embeddings = config.new_hf_embeddings()

# Avoiding unnecessary costs by caching the embeddings.
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace='hello-world'
)
