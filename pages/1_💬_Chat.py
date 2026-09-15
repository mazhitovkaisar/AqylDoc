import logging

import streamlit as st
from dotenv import load_dotenv

from src.rag_chain import answer_question
from src.utils import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AqylDoc — Chat", page_icon="💬")
st.title("💬 Chat")

# История чата хранится в session_state — Streamlit пересоздаёт скрипт при
# каждом действии, без этого сообщения пропадали бы после каждого ответа
if "messages" not in st.session_state:
    st.session_state.messages = []

# Перерисовываем всю прошлую историю сверху вниз при каждом rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Источники"):
                for i, chunk in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}] {chunk['metadata'].get('source')}**")
                    st.caption(chunk["text"][:400])

question = st.chat_input("Сұрағыңызды жазыңыз / Введите вопрос...")

# Новый вопрос: показываем его сразу, затем запускаем RAG-пайплайн
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # answer_question сама делает гибридный поиск и стримит ответ Claude;
            # write_stream печатает токены по мере поступления и возвращает полный текст
            token_stream, chunks = answer_question(question)
            answer = st.write_stream(token_stream)
        except RuntimeError as e:
            # Сюда попадаем, если не задан ANTHROPIC_API_KEY
            answer = f"⚠️ {e}"
            chunks = []
            st.error(answer)

        # Раскрывающийся список источников — какие фрагменты документов использовались
        if chunks:
            with st.expander("Источники"):
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**[{i}] {chunk['metadata'].get('source')}**")
                    st.caption(chunk["text"][:400])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": chunks}
    )
