from pathlib import Path

from config import DATA_DIR, configure_langsmith, make_embeddings, make_llm, require_keys

CONFIG = configure_langsmith()

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using only the provided context. "
            "If the context does not contain the answer, say you do not have enough information.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def build_vectorstore():
    require_keys(CONFIG)
    text = Path(DATA_DIR / "knowledge_base.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split knowledge base into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, make_embeddings(CONFIG))


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm(CONFIG)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {answer[:180]}\n")

    print(f"Sent {len(SAMPLE_QUESTIONS)} traces to LangSmith project '{CONFIG['langsmith_project']}'")
    print("Open https://smith.langchain.com to capture evidence/01_langsmith_traces.png")


if __name__ == "__main__":
    main()
