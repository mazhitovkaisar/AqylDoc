import io
import logging
import os
from pathlib import Path
from typing import Callable

import docx
import pytesseract
from PIL import Image
from pypdf import PdfReader

logger = logging.getLogger(__name__)

OCR_LANGS = "kaz+rus+eng"

# Если tesseract.exe не попал в PATH, путь к нему можно задать в .env (TESSERACT_CMD)
_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


# --- PDF: сначала пробуем текстовый слой, для страниц без текста запускаем OCR ---
def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if len(text.strip()) < 20:
            text += "\n" + _ocr_page_images(page)
        parts.append(text)
    return "\n".join(parts)


# OCR по картинкам, вложенным в страницу PDF (типичный случай для сканов)
def _ocr_page_images(page) -> str:
    text = ""
    for image_file in page.images:
        try:
            image = Image.open(io.BytesIO(image_file.data))
            text += "\n" + pytesseract.image_to_string(image, lang=OCR_LANGS)
        except Exception:
            logger.exception("OCR failed for an embedded image")
    return text


# --- DOCX: текст собирается из абзацев, пустые пропускаются ---
def load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


# --- Отдельные изображения (PNG/JPG) распознаются через OCR целиком ---
def load_image(path: Path) -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image, lang=OCR_LANGS)


# Диспетчер: выбирает нужный загрузчик по расширению файла
LOADERS: dict[str, Callable[[Path], str]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
}


def load_file(path: Path) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader(path)
