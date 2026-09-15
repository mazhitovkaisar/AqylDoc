import logging
import pickle
import re

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH

logger = logging.getLogger(__name__)

# Простая токенизация словами (без стемминга) — этого достаточно для точного
# совпадения ключевых терминов, чего не гарантирует семантический поиск
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# Индекс хранится как id + уже токенизированные тексты (pickle)
def _load() -> dict:
    if BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return {"ids": [], "tokenized": []}


def _save(data: dict) -> None:
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(data, f)


# Добавляет новые фрагменты в BM25-индекс (вызывается вместе с vectorstore.add_chunks)
def add_chunks(ids: list[str], texts: list[str]) -> None:
    data = _load()
    data["ids"].extend(ids)
    data["tokenized"].extend(_tokenize(t) for t in texts)
    _save(data)
    logger.info(f"Added {len(ids)} chunks to BM25 index")


# Убирает фрагменты удалённого документа из индекса
def delete_ids(ids_to_remove: set[str]) -> None:
    if not ids_to_remove:
        return
    data = _load()
    keep = [i for i, id_ in enumerate(data["ids"]) if id_ not in ids_to_remove]
    data["ids"] = [data["ids"][i] for i in keep]
    data["tokenized"] = [data["tokenized"][i] for i in keep]
    _save(data)


# Классический текстовый поиск по ключевым словам (без учёта смысла).
# BM25Okapi пересобирается на лету — нормально для учебного масштаба данных
def query(text: str, top_k: int) -> list[tuple[str, float]]:
    data = _load()
    if not data["ids"]:
        return []
    bm25 = BM25Okapi(data["tokenized"])
    scores = bm25.get_scores(_tokenize(text))
    ranked = sorted(zip(data["ids"], scores), key=lambda x: x[1], reverse=True)
    return [(id_, score) for id_, score in ranked[:top_k] if score > 0]
