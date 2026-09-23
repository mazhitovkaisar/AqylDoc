# AqylDoc

Учебный RAG-проект: AI-ассистент по личным документам (PDF, DOCX, сканы) на казахском и русском.

## Архитектура

```
Upload Documents → loaders.py (PDF/DOCX/OCR) → chunking.py
                 → embeddings.py (multilingual-e5-base, локально)
                 → vectorstore.py (NumPy, pickle) + bm25_index.py (BM25Okapi)

Chat → retrieval.py (гибридный поиск: RRF от вектора и BM25)
     → rag_chain.py (собирает контекст с источниками)
     → llm.py (Claude API / Ollama / Gemini — см. LLM_PROVIDER, потоковый ответ)
```

## Установка

1. Python-зависимости:
   ```
   pip install -r requirements.txt
   ```
2. Tesseract OCR (для сканов/изображений) — установите бинарник и языковые пакеты `kaz` и `rus`:
   https://github.com/UB-Mannheim/tesseract/wiki (Windows-сборка с выбором языков при установке).
3. Скопируйте `.env.example` в `.env` и настройте `LLM_PROVIDER`:
   - `ollama` — бесплатно, локально. Установите [Ollama](https://ollama.com), выполните
     `ollama pull qwen2.5:7b` (или другую модель — впишите её имя в `OLLAMA_MODEL`),
     запустите приложение Ollama перед стартом AqylDoc.
   - `anthropic` — платно, лучшее качество на казахском/русском. Впишите
     `ANTHROPIC_API_KEY` с https://console.anthropic.com (нужен баланс на счету).
   - `gemini` — бесплатно и работает в облаке (в отличие от Ollama). Получите
     `GEMINI_API_KEY` на https://aistudio.google.com/apikey — привязка карты не нужна.

## Запуск

```
streamlit run Welcome.py
```

Откроется браузер с тремя страницами: **Welcome**, **Chat**, **Upload Documents**.

## Деплой (Streamlit Community Cloud)

1. Зайти на https://share.streamlit.io, войти через GitHub, выбрать этот репозиторий,
   ветка `main`, главный файл `Welcome.py`.
2. В Advanced settings → Secrets указать (вариант с Claude, платно):
   ```
   ANTHROPIC_API_KEY = "..."
   CLAUDE_MODEL = "claude-sonnet-5"
   EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
   ```
   или бесплатный вариант с Gemini:
   ```
   LLM_PROVIDER = "gemini"
   GEMINI_API_KEY = "..."
   EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
   ```
   Модель эмбеддингов для облака взята меньше локальной (`e5-base` → `e5-small`),
   так как бесплатный тариф даёт всего 1 ГБ RAM, а `e5-base` сама весит в памяти больше.
3. `packages.txt` в репозитории ставит системный Tesseract с казахским/русским —
   ничего дополнительно делать не нужно.

**Ограничение:** файловая система на бесплатном тарифе не гарантированно постоянна —
загруженные документы и индекс могут не пережить перезапуск/засыпание приложения.
Для демо это ок, для реального использования нужно внешнее хранилище.

## Известные ограничения

- Векторный поиск и BM25-индекс перестраиваются в памяти при каждом запросе — нормально
  для сотен/тысяч фрагментов (учебный масштаб), для продакшена нужны persistent-сервисы.
- OCR работает только там, где страница PDF — это одно вложенное изображение
  (типичный случай для сканов); PDF, свёрстанные из множества мелких картинок, не поддержаны.
