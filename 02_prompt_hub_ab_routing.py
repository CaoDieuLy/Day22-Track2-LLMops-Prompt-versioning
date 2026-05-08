import hashlib

from config import configure_langsmith, make_embeddings, make_llm, require_keys

CONFIG = configure_langsmith()

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS


SYSTEM_V1 = (
    "You are a concise RAG assistant. Answer using only the context below. "
    "Use 2-4 sentences. If the answer is not in context, say you do not have enough information.\n\n"
    "Context:\n{context}"
)
SYSTEM_V2 = (
    "You are an expert AI tutor. Use only the context below.\n\n"
    "Instructions:\n"
    "1. Identify the facts relevant to the question.\n"
    "2. Give a structured, accurate answer in 3-5 sentences.\n"
    "3. State clearly when the context lacks enough information.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])
PROMPT_V1_NAME = "day22-rag-prompt-v1"
PROMPT_V2_NAME = "day22-rag-prompt-v2"


def push_prompts_to_hub(client):
    for name, prompt, description in [
        (PROMPT_V1_NAME, PROMPT_V1, "Day22 RAG prompt V1 - concise answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "Day22 RAG prompt V2 - structured tutor answers"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"Pushed {name}: {url}")
        except Exception as exc:
            print(f"Could not push {name}: {exc}")


def pull_prompts_from_hub(client):
    prompts = {}
    for name, fallback in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"Pulled {name} from Prompt Hub")
        except Exception as exc:
            prompts[name] = fallback
            print(f"Using local fallback for {name}: {exc}")
    return prompts


def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


def build_vectorstore():
    from pathlib import Path
    from config import DATA_DIR

    require_keys(CONFIG)
    text = Path(DATA_DIR / "knowledge_base.txt").read_text(encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    return FAISS.from_texts(chunks, make_embeddings(CONFIG))


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version, "context": context}


def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    require_keys(CONFIG)
    client = Client(api_key=CONFIG["langsmith_api_key"])
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm(CONFIG)
    counts = {"v1": 0, "v2": 0}

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        counts[version_tag] += 1
        result = ask_ab(retriever, llm, prompts[version_key], question, version_tag)
        print(f"[{i + 1:02d}] [prompt-{version_tag}] {question}")
        print(f"     {result['answer'][:160]}")

    print(f"Routing summary: prompt-v1={counts['v1']}, prompt-v2={counts['v2']}")
    print("Capture evidence/02_prompt_hub.png and save this output to evidence/02_ab_routing_log.txt")


if __name__ == "__main__":
    main()
