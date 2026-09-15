# AqylDoc

Учебный RAG-проект: AI-ассистент по личным документам (PDF, DOCX, сканы) на казахском и русском.

## Архитектура

```
Upload Documents → loaders.py (PDF/DOCX/OCR) → chunking.py
                 → embeddings.py (multilingual-e5-base, локально)
                 → vectorstore.py (NumPy, pickle) + bm25_index.py (BM25Okapi)

Chat → retrieval.py (гибридный поиск: RRF от вектора и BM25)
     → rag_chain.py (собирает контекст с источниками)
     → llm.py (Claude API, потоковый ответ)
```

## Установка

1. Python-зависимости:
   ```
   pip install -r requirements.txt
   ```
2. Tesseract OCR (для сканов/изображений) — установите бинарник и языковые пакеты `kaz` и `rus`:
   https://github.com/UB-Mannheim/tesseract/wiki (Windows-сборка с выбором языков при установке).
3. Скопируйте `.env.example` в `.env` и впишите свой `ANTHROPIC_API_KEY`.

## Запуск

```
streamlit run Welcome.py
```

Откроется браузер с тремя страницами: **Welcome**, **Chat**, **Upload Documents**.

## Известные ограничения

- Векторный поиск и BM25-индекс перестраиваются в памяти при каждом запросе — нормально
  для сотен/тысяч фрагментов (учебный масштаб), для продакшена нужны persistent-сервисы.
- OCR работает только там, где страница PDF — это одно вложенное изображение
  (типичный случай для сканов); PDF, свёрстанные из множества мелких картинок, не поддержаны.
