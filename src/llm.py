import logging
import os
from typing import Iterator

import anthropic

from src.config import CLAUDE_MODEL

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


# Потоковая генерация ответа — токены приходят по мере готовности,
# в интерфейсе это даёт эффект "печатающегося" текста (st.write_stream)
def stream_answer(system_prompt: str, user_message: str) -> Iterator[str]:
    client = get_client()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        yield from stream.text_stream
