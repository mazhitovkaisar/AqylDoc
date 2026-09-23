import hashlib
import logging
from pathlib import Path

from src import bm25_index, vectorstore
from src.chunking import chunk_text
from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from src.embeddings import embed_passages
from src.loaders import load_file

logger = logging.getLogger(__name__)


# Полный путь одного файла в индекс: текст -> чанки -> эмбеддинги -> запись
# в оба индекса (вектор + BM25) с одинаковыми id, чтобы они не разъезжались
def ingest_file(path: Path) -> int:
    text = load_file(path)
    chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    if not chunks:
        logger.warning(f"No extractable text in {path.name}")
        return 0

    # id строится из имени файла + номера чанка + хэша текста — стабилен между запусками
    ids = [
        f"{path.name}::{i}::{hashlib.md5(c.encode()).hexdigest()[:8]}"
        for i, c in enumerate(chunks)
    ]
    metadatas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]
    # Эмбеддинги считаем до удаления старых чанков: если модель/RAM падает,
    # уже проиндексированный документ не пропадает.
    embeddings = embed_passages(chunks)

    # Повторная загрузка того же файла не должна дублировать чанки: старые
    # фрагменты этого источника убираем до записи новых (иначе hybrid search
    # получает одинаковые id несколько раз и забивает TOP_K дубликатами).
    delete_document(path.name)

    vectorstore.add_chunks(ids, chunks, embeddings, metadatas)
    bm25_index.add_chunks(ids, chunks)
    logger.info(f"Ingested {len(chunks)} chunks from {path.name}")
    return len(chunks)


# Удаляет документ из обоих индексов разом (источник правды по id — vectorstore)
def delete_document(source: str) -> None:
    removed_ids = vectorstore.delete_source(source)
    bm25_index.delete_ids(removed_ids)
    logger.info(f"Deleted document {source} ({len(removed_ids)} chunks)")
