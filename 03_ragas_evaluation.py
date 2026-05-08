import json
import warnings
from pathlib import Path

import numpy as np

from config import DATA_DIR, configure_langsmith, make_embeddings, make_llm, require_keys
from qa_pairs import QA_PAIRS

warnings.filterwarnings("ignore")
CONFIG = configure_langsmith()

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from langsmith import traceable


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
PROMPTS = {
    "v1": ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")]),
    "v2": ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")]),
}


def build_vectorstore():
    require_keys(CONFIG)
    text = Path(DATA_DIR / "knowledge_base.txt").read_text(encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    return FAISS.from_texts(chunks, make_embeddings(CONFIG))


@traceable(name="ragas-rag-query", tags=["ragas", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": "\n\n".join(contexts), "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm(CONFIG)
    prompt = PROMPTS[prompt_version]
    results = []

    print(f"\nRunning {len(QA_PAIRS)} questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question']}")
    return results


def build_ragas_dataset(rag_results: list):
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=make_llm(CONFIG),
        embeddings=make_embeddings(CONFIG),
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        values = [v for v in raw if v is not None and not np.isnan(v)]
        scores[key] = float(np.mean(values)) if values else 0.0
        mark = " target" if key == "faithfulness" and scores[key] >= 0.8 else ""
        print(f"  {key:30s}: {scores[key]:.4f}{mark}")
    return scores


def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\nComparison")
    print("-" * 70)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2"
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  winner={winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    target_met = best_faith >= 0.8
    print(f"\n{'Target met' if target_met else 'Below target'}: faithfulness = {best_faith:.4f}")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": target_met,
        "best_faithfulness": best_faith,
    }
    Path(DATA_DIR / "ragas_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved data/ragas_report.json")
    print("Capture this table as evidence/03_ragas_scores.png")


if __name__ == "__main__":
    main()
