import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


# Модель грузится в память один раз за процесс (lru_cache) — это и есть
# та самая пауза при первом обращении, дальше эмбеддинги считаются быстро
@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


# Эмбеддинги для фрагментов документов при индексации (префикс "passage:" —
# требование модели e5 для несимметричного поиска запрос↔документ)
def embed_passages(texts: list[str]) -> np.ndarray:
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(prefixed, normalize_embeddings=True)


# Эмбеддинг вопроса пользователя перед поиском (префикс "query:")
def embed_query(text: str) -> np.ndarray:
    model = get_model()
    return model.encode(f"query: {text}", normalize_embeddings=True)
