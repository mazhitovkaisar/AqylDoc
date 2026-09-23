import os
from pathlib import Path

# --- Пути проекта и папки с данными (создаются автоматически при первом запуске) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_PATH = DATA_DIR / "vector_store.pkl"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
LOG_DIR = BASE_DIR / "logs"

for d in (DATA_DIR, UPLOADS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Модели: эмбеддинги считаются локально; ответы генерирует LLM ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# LLM_PROVIDER переключает генерацию ответа:
# - "anthropic" — Claude API, платно, лучшее качество на казахском/русском
# - "ollama"    — бесплатно, только локально (в облаке негде держать процесс модели)
# - "gemini"    — Google Gemini, бесплатный тариф, работает и в облаке
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Разбивка документов на фрагменты (в словах, с перекрытием) ---
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# --- Сколько фрагментов брать у каждого поисковика и сколько оставить после слияния ---
TOP_K_VECTOR = 8
TOP_K_BM25 = 8
TOP_K_FINAL = 5
