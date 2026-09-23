import logging
import os
from typing import Iterator

import anthropic
import ollama
from google import genai
from google.genai import types as genai_types

from src.config import CLAUDE_MODEL, GEMINI_MODEL, LLM_PROVIDER, OLLAMA_MODEL

logger = logging.getLogger(__name__)

_client = None
_gemini_client = None


def _clean_api_key(env_var: str) -> str:
    key = (os.environ.get(env_var) or "").strip()
    if not key:
        raise RuntimeError(f"{env_var} is not set. Add it to your .env file.")
    if not key.isascii():
        # Обычно значит, что при вставке ключа в Secrets/.env затесался
        # невидимый символ (неразрывный пробел, смарт-кавычка и т.п.) —
        # httpx падает с криптичным UnicodeEncodeError при сборке заголовка
        raise RuntimeError(
            f"{env_var} содержит недопустимые символы (похоже, при копировании "
            f"попал невидимый символ). Скопируйте ключ заново и вставьте как обычный текст."
        )
    return key


# Клиент Claude создаётся один раз за процесс и переиспользуется
def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=_clean_api_key("ANTHROPIC_API_KEY"))
    return _client


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=_clean_api_key("GEMINI_API_KEY"))
    return _gemini_client


def _stream_anthropic(system_prompt: str, user_message: str) -> Iterator[str]:
    client = get_client()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        yield from stream.text_stream


def _stream_gemini(system_prompt: str, user_message: str) -> Iterator[str]:
    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=user_message,
        config=genai_types.GenerateContentConfig(system_instruction=system_prompt),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


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
    if LLM_PROVIDER == "gemini":
        return _stream_gemini(system_prompt, user_message)
    return _stream_anthropic(system_prompt, user_message)
