import json
import time
from pathlib import Path

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (Faithfulness,LLMContextPrecisionWithoutReference,ResponseRelevancy)
from ragas.run_config import RunConfig

from src.config import (EMBEDDING_MODEL,GROQ_API_KEY,OLLAMA_BASE_URL,RERANK_TOP_N,RETRIEVAL_TOP_K,get_model_name)
from src.rag_chain import run_rag_pipeline
from src.retriever import retrieve_documents
from src.schemas import QueryRequest

DATASET_PATH = Path("eval/test_dataset.json")
RESULTS_PATH = Path("eval/ragas_results.json")


def main() -> None:
    eval_cases: list[dict[str, str]] = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    rows: list[dict] = []
    for idx, case in enumerate(eval_cases, start=1):
        question = case["question"]
        source_file = case["source_file"]
        print(f"[{idx}/{len(eval_cases)}] Running RAG pipeline on: {question}")

        resp = run_rag_pipeline(
            QueryRequest(
                query=question,
                source_file=source_file,
                top_n=RERANK_TOP_N,
            )
        )

        docs = retrieve_documents(
            query=question,
            source_file=source_file,
            initial_k=RETRIEVAL_TOP_K,
            top_n=RERANK_TOP_N,
        )
        full_contexts = [doc.page_content for doc in docs]

        rows.append(
            {
                "user_input": question,
                "response": resp.answer,
                "retrieved_contexts": full_contexts,
            }
        )

        if idx < len(eval_cases):
            time.sleep(15)

    eval_dataset = Dataset.from_list(rows)

    judge_model_name = get_model_name(role="fast", provider="groq")
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=judge_model_name,
            api_key=GROQ_API_KEY,
            temperature=0,max_tokens=4096,reasoning_effort='low'
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings,strictness=1),
        LLMContextPrecisionWithoutReference(llm=judge_llm),
    ]

    print("\nCooling down 65s to reset Groq TPM bucket before Ragas judging...")
    time.sleep(65)

    print(f"Running Ragas evaluation sequentially using judge: {judge_model_name}...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=RunConfig(
            max_workers=1,
            timeout=180,
            max_retries=5,
            max_wait=60,
        ),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    print("\n=== Ragas Evaluation Results ===")
    display_cols = [
        col
        for col in [
            "user_input",
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision_without_reference",
        ]
        if col in df.columns
    ]
    print(df[display_cols])

    RESULTS_PATH.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"\nSaved full metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()