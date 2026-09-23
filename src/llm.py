import logging
import os
from typing import Iterator

import anthropic
import ollama

from src.config import CLAUDE_MODEL, LLM_PROVIDER, OLLAMA_MODEL

logger = logging.getLogger(__name__)

_client = None


# Клиент Claude создаётся один раз за процесс и переиспользуется
def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _stream_anthropic(system_prompt: str, user_message: str) -> Iterator[str]:
    client = get_client()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        yield from stream.text_stream


def _stream_ollama(system_prompt: str, user_message: str) -> Iterator[str]:
    try:
        chunks = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for chunk in chunks:
            yield chunk["message"]["content"]
    except Exception as e:
        # Самая частая причина — приложение Ollama не запущено или модель не скачана
        raise RuntimeError(
            f"Не удалось получить ответ от Ollama (модель {OLLAMA_MODEL}). "
            f"Убедитесь, что приложение Ollama запущено и модель загружена "
            f"(ollama pull {OLLAMA_MODEL}). Исходная ошибка: {e}"
        ) from e


# Потоковая генерация ответа — токены приходят по мере готовности,
# в интерфейсе это даёт эффект "печатающегося" текста (st.write_stream).
# Провайдер выбирается через LLM_PROVIDER в .env
def stream_answer(system_prompt: str, user_message: str) -> Iterator[str]:
    if LLM_PROVIDER == "ollama":
        return _stream_ollama(system_prompt, user_message)
    return _stream_anthropic(system_prompt, user_message)
