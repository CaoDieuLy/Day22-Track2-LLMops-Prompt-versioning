import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EVIDENCE_DIR = ROOT / "evidence"


def load_config() -> dict:
    load_dotenv(ROOT / ".env")

    langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")

    config = {
        "langsmith_api_key": langsmith_key,
        "langsmith_project": os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "day22-langsmith-rag",
        "langsmith_endpoint": os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com",
        "openai_api_key": openai_key,
        "openai_base_url": openai_base_url,
        "llm_model": os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-5.4-mini",
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
    }
    return config


def configure_langsmith() -> dict:
    config = load_config()
    if config["langsmith_api_key"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config["langsmith_api_key"]
        os.environ["LANGCHAIN_PROJECT"] = config["langsmith_project"]
        os.environ["LANGCHAIN_ENDPOINT"] = config["langsmith_endpoint"]
    return config


def require_keys(config: dict, *, langsmith: bool = True, openai: bool = True) -> None:
    missing = []
    if langsmith and not config["langsmith_api_key"]:
        missing.append("LANGSMITH_API_KEY or LANGCHAIN_API_KEY")
    if openai and not config["openai_api_key"]:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))


def make_llm(config: dict, temperature: float | None = None):
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": config["llm_model"],
        "api_key": config["openai_api_key"],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if config["openai_base_url"]:
        kwargs["base_url"] = config["openai_base_url"]
    return ChatOpenAI(**kwargs)


def make_embeddings(config: dict):
    from langchain_openai import OpenAIEmbeddings

    kwargs = {
        "model": config["embedding_model"],
        "api_key": config["openai_api_key"],
        "check_embedding_ctx_length": False,
    }
    if config["openai_base_url"]:
        kwargs["base_url"] = config["openai_base_url"]
    return OpenAIEmbeddings(**kwargs)


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded")
    print(f"   LangSmith project : {cfg['langsmith_project']}")
    print(f"   OpenAI endpoint   : {cfg['openai_base_url'] or 'default OpenAI endpoint'}")
    print(f"   Default LLM model : {cfg['llm_model']}")
    print(f"   Embedding model   : {cfg['embedding_model']}")
    print(f"   LangSmith key     : {'set' if cfg['langsmith_api_key'] else 'missing'}")
    print(f"   OpenAI key        : {'set' if cfg['openai_api_key'] else 'missing'}")
