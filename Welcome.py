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
Жеке құжаттарыңыз бойынша сұрақ қойыңыз — AqylDoc қазақша және орысша жауап береді.
Задайте вопрос по своим документам — AqylDoc отвечает на казахском и русском.

**Как это работает:**
1. Загрузите PDF, DOCX или изображения на странице **Upload Documents**.
2. Система разбивает их на фрагменты, считает эмбеддинги и индексирует локально
   (векторный поиск + BM25 — гибридный поиск: смысл + точные термины).
3. На странице **Chat** ваш вопрос ищет релевантные фрагменты и отправляется вместе
   с ними в Claude, который отвечает на языке вопроса со ссылками на источники.

**Выберите страницу слева, чтобы начать.**
"""
)