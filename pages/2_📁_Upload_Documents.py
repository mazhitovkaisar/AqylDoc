import logging

import streamlit as st
from dotenv import load_dotenv

from src.config import UPLOADS_DIR
from src.ingest_pipeline import delete_document, ingest_file
from src.utils import setup_logging
from src.vectorstore import list_sources

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AqylDoc — Upload", page_icon="📁")
st.title("📁 Upload Documents")

# Выбор файлов для загрузки (индексация запускается отдельной кнопкой ниже)
uploaded_files = st.file_uploader(
    "PDF, DOCX немесе сурет (PNG/JPG) жүктеңіз",
    type=["pdf", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# Сохраняем файл на диск и прогоняем через ingest_file (загрузка -> чанки -> эмбеддинги -> индекс)
if uploaded_files and st.button("Индекстеу / Индексировать"):
    progress = st.progress(0.0)
    for i, uploaded_file in enumerate(uploaded_files):
        dest = UPLOADS_DIR / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())
        with st.spinner(f"Обработка {uploaded_file.name}..."):
            try:
                n_chunks = ingest_file(dest)
                if n_chunks:
                    st.success(f"{uploaded_file.name}: {n_chunks} фрагментов проиндексировано")
                else:
                    st.warning(f"{uploaded_file.name}: текст не найден")
            except Exception as e:
                logger.exception(f"Failed to ingest {uploaded_file.name}")
                st.error(f"{uploaded_file.name}: ошибка — {e}")
        progress.progress((i + 1) / len(uploaded_files))

st.divider()
st.subheader("Проиндексированные документы")

# Список источников берётся напрямую из vectorstore; кнопка "Удалить" чистит оба индекса
sources = list_sources()
if not sources:
    st.caption("Пока нет загруженных документов.")
else:
    for source in sources:
        col1, col2 = st.columns([4, 1])
        col1.write(source)
        if col2.button("Удалить", key=f"del_{source}"):
            delete_document(source)
            st.rerun()
