from typing import Iterator

from src.llm import stream_answer
from src.retrieval import hybrid_search

# Инструкция для Claude: отвечать только по найденному контексту, на языке вопроса,
# честно признавать нехватку данных и ссылаться на источники маркерами [n]
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the document \
excerpts given below as context. Respond in the same language as the user's question \
(Kazakh or Russian). If the context does not contain the answer, say so honestly instead \
of guessing. Cite sources inline using [n] markers matching the numbered entries below.

Context:
{context}
"""


# Собирает найденные фрагменты в пронумерованный блок текста для системного промпта
def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{chunk['text']}")
    return "\n\n".join(parts)


# Полный RAG-цикл: гибридный поиск -> контекст -> потоковый ответ Claude.
# Возвращает и поток токенов, и сами чанки — чтобы показать источники в интерфейсе
def answer_question(question: str) -> tuple[Iterator[str], list[dict]]:
    chunks = hybrid_search(question)
    context = build_context(chunks) if chunks else "No relevant context was found."
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return stream_answer(system_prompt, question), chunks
