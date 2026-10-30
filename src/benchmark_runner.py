import os
from collections import defaultdict

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from langchain_core.documents import Document

from .config import DEFAULT_EMBED_MODEL
from .kb import build_knowledge_base
from .rag_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    mrr_at_k,
)


def download_beir_dataset(dataset_name: str, data_dir: str = "datasets"):
    """
    Download a BEIR dataset if not already downloaded.
    Default dataset is SciFact.
    """
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(out_dir):
        print(f"Downloading BEIR dataset: {dataset_name}")
        data_path = util.download_and_unzip(url, data_dir)
    else:
        print(f"BEIR dataset already downloaded: {dataset_name}")
        data_path = out_dir

    return data_path


def load_beir_dataset(dataset_name: str, split: str = "test"):
    """
    BEIR format:
    corpus = documents
    queries = questions/claims
    qrels = gold relevant document labels
    """
    data_path = download_beir_dataset(dataset_name)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    return corpus, queries, qrels


def make_small_subset(corpus, queries, qrels, max_docs=500, max_queries=25):
    """
    Make a small benchmark subset for fast testing.

    Important:
    We make sure selected queries still have gold documents
    inside the selected corpus.
    """
    if max_docs is None and max_queries is None:
        return corpus, queries, qrels

    selected_qrels = {}
    selected_queries = {}
    required_doc_ids = set()

    for query_id, gold_dict in qrels.items():
        if query_id not in queries:
            continue

        positive_gold = {
            doc_id: rel
            for doc_id, rel in gold_dict.items()
            if rel > 0 and doc_id in corpus
        }

        if not positive_gold:
            continue

        selected_qrels[query_id] = positive_gold
        selected_queries[query_id] = queries[query_id]
        required_doc_ids.update(positive_gold.keys())

        if max_queries is not None and len(selected_queries) >= max_queries:
            break

    selected_corpus = {}

    for doc_id in required_doc_ids:
        selected_corpus[doc_id] = corpus[doc_id]

    if max_docs is not None:
        for doc_id, item in corpus.items():
            if len(selected_corpus) >= max_docs:
                break

            if doc_id not in selected_corpus:
                selected_corpus[doc_id] = item

    return selected_corpus, selected_queries, selected_qrels


def corpus_to_documents(corpus):
    """
    Convert BEIR corpus dictionary into LangChain Documents.

    Every document keeps its original BEIR doc_id.
    After chunking, each chunk still has the parent doc_id.
    """
    docs = []

    for doc_id, item in corpus.items():
        title = item.get("title", "")
        text = item.get("text", "")

        full_text = f"{title}\n\n{text}".strip()

        docs.append(
            Document(
                page_content=full_text,
                metadata={
                    "doc_id": doc_id,
                    "title": title,
                    "source": f"beir:{doc_id}",
                    "doc_type": "beir",
                },
            )
        )

    return docs


def retrieve_doc_ids(vector_store, query, k):
    """
    Retrieve chunks from the vector store, then convert those chunks
    back to parent document IDs for BEIR qrels scoring.
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    retrieved_ids = []

    for doc, score in results:
        retrieved_ids.append(doc.metadata["doc_id"])

    # Remove duplicate doc IDs caused by chunking.
    deduped = []
    seen = set()

    for doc_id in retrieved_ids:
        if doc_id not in seen:
            deduped.append(doc_id)
            seen.add(doc_id)

    return deduped


def evaluate_retrieval(vector_store, queries, qrels, k_values):
    """
    Retrieval-only evaluation.

    q = one BEIR/SciFact query
    r_k(q) = top-k retrieved document IDs
    G(q) = gold relevant document IDs from qrels
    """
    totals = {
        k: defaultdict(float)
        for k in k_values
    }

    count = 0

    for query_id, gold_dict in qrels.items():
        if query_id not in queries:
            continue

        q = queries[query_id]

        gold_ids = [
            doc_id
            for doc_id, relevance in gold_dict.items()
            if relevance > 0
        ]

        if not gold_ids:
            continue

        # Retrieve extra chunks so we still have enough unique doc IDs
        # after deduplication.
        retrieve_k = max(k_values) * 5

        retrieved_ids = retrieve_doc_ids(
            vector_store=vector_store,
            query=q,
            k=retrieve_k,
        )

        for k in k_values:
            precision = precision_at_k(retrieved_ids, gold_ids, k)
            recall = recall_at_k(retrieved_ids, gold_ids, k)
            f1 = f1_at_k(precision, recall)
            hit_rate = hit_rate_at_k(retrieved_ids, gold_ids, k)
            mrr = mrr_at_k(retrieved_ids, gold_ids, k)

            totals[k]["precision"] += precision
            totals[k]["recall"] += recall
            totals[k]["f1"] += f1
            totals[k]["hit_rate"] += hit_rate
            totals[k]["mrr"] += mrr

        count += 1

        if count % 25 == 0:
            print(f"Evaluated {count} queries...")

    print(f"\nTotal evaluated queries: {count}")

    for k in k_values:
        print(f"\nScores @ {k}")
        print("-" * 35)

        if count == 0:
            print("No valid queries evaluated.")
            continue

        print(f"Precision@{k}: {totals[k]['precision'] / count:.4f}")
        print(f"Recall@{k}:    {totals[k]['recall'] / count:.4f}")
        print(f"F1@{k}:        {totals[k]['f1'] / count:.4f}")
        print(f"HitRate@{k}:   {totals[k]['hit_rate'] / count:.4f}")
        print(f"MRR@{k}:       {totals[k]['mrr'] / count:.4f}")


def run_benchmark(
    dataset_name="scifact",
    split="test",
    embed_model=DEFAULT_EMBED_MODEL,
    max_docs=500,
    max_queries=25,
    full_benchmark=False,
):
    """
    Run BEIR benchmark mode using the same vector-store builder as the main RAG system.
    """
    print("\nStarting vector-RAG benchmark mode")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Embedding model: {embed_model}")

    corpus, queries, qrels = load_beir_dataset(
        dataset_name=dataset_name,
        split=split,
    )

    print(f"\nOriginal corpus documents: {len(corpus)}")
    print(f"Original queries: {len(queries)}")
    print(f"Original qrels: {len(qrels)}")

    if full_benchmark:
        max_docs = None
        max_queries = None
        print("\nRunning FULL benchmark.")
    else:
        print(f"\nRunning small benchmark subset: max_docs={max_docs}, max_queries={max_queries}")

    corpus, queries, qrels = make_small_subset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        max_docs=max_docs,
        max_queries=max_queries,
    )

    print(f"\nBenchmark corpus documents: {len(corpus)}")
    print(f"Benchmark queries: {len(queries)}")
    print(f"Benchmark qrels: {len(qrels)}")

    docs = corpus_to_documents(corpus)

    vector_store = build_knowledge_base(
        docs=docs,
        embed_model=embed_model,
    )

    evaluate_retrieval(
        vector_store=vector_store,
        queries=queries,
        qrels=qrels,
        k_values=[1, 3, 5, 10],
    )