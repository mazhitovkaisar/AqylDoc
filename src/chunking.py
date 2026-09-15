# Режет текст документа на перекрывающиеся фрагменты фиксированной длины (в словах).
# Перекрытие (overlap) нужно, чтобы смысл не обрывался ровно на границе чанка.
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks
