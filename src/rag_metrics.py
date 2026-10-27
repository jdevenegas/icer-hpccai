def precision_at_k(retrieved_ids, gold_ids, k):
    """
    Precision@k(q) = |r_k(q) ∩ G(q)| / |r_k(q)|
    """
    retrieved_k = retrieved_ids[:k]
    gold_ids = set(gold_ids)

    if len(retrieved_k) == 0:
        return 0.0

    hits = sum(1 for doc_id in retrieved_k if doc_id in gold_ids)
    return hits / len(retrieved_k)


def recall_at_k(retrieved_ids, gold_ids, k):
    """
    Recall@k(q) = |r_k(q) ∩ G(q)| / |G(q)|
    """
    retrieved_k = set(retrieved_ids[:k])
    gold_ids = set(gold_ids)

    if len(gold_ids) == 0:
        return 0.0

    hits = len(retrieved_k.intersection(gold_ids))
    return hits / len(gold_ids)


def f1_at_k(precision, recall):
    """
    F1@k(q) = 2PR / (P + R)
    """
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(retrieved_ids, gold_ids, k):
    """
    HitRate@k(q) = 1 if at least one retrieved unit is relevant, else 0.
    """
    retrieved_k = set(retrieved_ids[:k])
    gold_ids = set(gold_ids)

    return 1.0 if retrieved_k.intersection(gold_ids) else 0.0


def mrr_at_k(retrieved_ids, gold_ids, k):
    """
    MRR@k(q) = 1 / rank of first relevant result.
    """
    gold_ids = set(gold_ids)

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / rank

    return 0.0