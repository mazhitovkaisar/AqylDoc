import logging
import streamlit as st
from dotenv import load_dotenv
from src.utils import setup_logging
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
st.set_page_config(page_title="AqylDoc", page_icon="📚")
# Главная страница — только описание проекта, вся логика в pages
st.title("AqylDoc 📚")
st.markdown(
    """
Құжаттарыңызды жүктеп, сұрақ қойыңыз — AqylDoc қазақша немесе орысша жауап береді.
Загрузите документы и задайте вопрос — AqylDoc ответит на казахском или русском.

**Как пользоваться:**
1. Откройте страницу **Upload Documents** и загрузите свои файлы (PDF, DOCX, фото).
2. Перейдите на страницу **Chat** и напишите вопрос.
3. Ответ придёт на том языке, на котором вы спросили — на казахском или русском.

**Выберите страницу слева, чтобы начать.**
"""
)