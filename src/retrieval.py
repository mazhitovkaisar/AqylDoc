import logging

from src import bm25_index, vectorstore
from src.config import TOP_K_BM25, TOP_K_FINAL, TOP_K_VECTOR
from src.embeddings import embed_query

logger = logging.getLogger(__name__)

RRF_K = 60


def hybrid_search(query_text: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """Fuses vector search and BM25 rankings with reciprocal rank fusion,
    so a chunk that ranks well in either signal surfaces even if it's weak in the other."""
    # Два независимых поиска: по смыслу (вектор) и по точным словам (BM25)
    query_embedding = embed_query(query_text)
    vector_hits = vectorstore.query(query_embedding, TOP_K_VECTOR)
    bm25_hits = bm25_index.query(query_text, TOP_K_BM25)

    # RRF: каждому чанку начисляется очки по его позиции (рангу) в каждом списке,
    # а не по абсолютному значению score — так вектор и BM25 сравнимы между собой
    scores: dict[str, float] = {}
    for rank, hit in enumerate(vector_hits):
        scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (RRF_K + rank + 1)
    for rank, (id_, _) in enumerate(bm25_hits):
        scores[id_] = scores.get(id_, 0.0) + 1 / (RRF_K + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    documents = vectorstore.get_by_ids(ranked_ids)

    return [
        {"id": id_, "text": documents[id_]["text"], "metadata": documents[id_]["metadata"]}
        for id_ in ranked_ids
        if id_ in documents
    ]
