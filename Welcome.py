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

**Қалай пайдалану керек / Как пользоваться:**
1. **Upload Documents** бетін ашып, файлдарыңызды жүктеңіз (PDF, DOCX, фото).
   Откройте страницу **Upload Documents** и загрузите свои файлы (PDF, DOCX, фото).
2. **Chat** бетіне өтіп, сұрағыңызды жазыңыз.
   Перейдите на страницу **Chat** и напишите вопрос.
3. Жауап сіз сұраған тілде келеді — қазақша немесе орысша.
   Ответ придёт на том языке, на котором вы спросили — на казахском или русском.

**Бастау үшін сол жақтан бетті таңдаңыз / Выберите страницу слева, чтобы начать.**
"""
)