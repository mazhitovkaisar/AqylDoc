import logging
import pickle

import numpy as np

from src.config import VECTOR_STORE_PATH

logger = logging.getLogger(__name__)


# Всё хранилище — один pickle-файл: список id, текстов, метаданных и матрица эмбеддингов
def _load() -> dict:
    if VECTOR_STORE_PATH.exists():
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return {"ids": [], "texts": [], "metadatas": [], "embeddings": None}


def _save(data: dict) -> None:
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(data, f)


# Добавляет новые фрагменты и их эмбеддинги в хранилище (вызывается при индексации документа)
def add_chunks(
    ids: list[str], texts: list[str], embeddings: np.ndarray, metadatas: list[dict]
) -> None:
    data = _load()
    data["ids"].extend(ids)
    data["texts"].extend(texts)
    data["metadatas"].extend(metadatas)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    data["embeddings"] = (
        embeddings if data["embeddings"] is None else np.vstack([data["embeddings"], embeddings])
    )
    _save(data)
    logger.info(f"Added {len(ids)} chunks to vector store")


# Семантический поиск: косинусное сходство = скалярное произведение,
# т.к. эмбеддинги уже нормализованы (normalize_embeddings=True)
def query(embedding: np.ndarray, top_k: int) -> list[dict]:
    data = _load()
    if data["embeddings"] is None or not data["ids"]:
        return []
    scores = data["embeddings"] @ np.asarray(embedding, dtype=np.float32)
    top_idx = np.argsort(-scores)[:top_k]
    return [
        {
            "id": data["ids"][i],
            "text": data["texts"][i],
            "metadata": data["metadatas"][i],
            "score": float(scores[i]),
        }
        for i in top_idx
    ]


# Достаёт текст и метаданные по конкретным id (используется гибридным поиском)
def get_by_ids(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    data = _load()
    wanted = set(ids)
    return {
        id_: {"text": text, "metadata": meta}
        for id_, text, meta in zip(data["ids"], data["texts"], data["metadatas"])
        if id_ in wanted
    }


# Список уникальных имён файлов, уже проиндексированных в хранилище (для страницы Upload)
def list_sources() -> list[str]:
    data = _load()
    sources = {m.get("source") for m in data["metadatas"] if m.get("source")}
    return sorted(sources)


# Удаляет все фрагменты одного документа; возвращает их id, чтобы почистить и BM25-индекс тоже
def delete_source(source: str) -> set[str]:
    data = _load()
    keep_idx = [i for i, m in enumerate(data["metadatas"]) if m.get("source") != source]
    if len(keep_idx) == len(data["ids"]):
        return set()

    keep_set = set(keep_idx)
    removed_ids = {id_ for i, id_ in enumerate(data["ids"]) if i not in keep_set}

    data["ids"] = [data["ids"][i] for i in keep_idx]
    data["texts"] = [data["texts"][i] for i in keep_idx]
    data["metadatas"] = [data["metadatas"][i] for i in keep_idx]
    data["embeddings"] = data["embeddings"][keep_idx] if data["embeddings"] is not None else None
    _save(data)
    return removed_ids
