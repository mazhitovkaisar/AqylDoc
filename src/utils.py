import logging

from src.config import LOG_DIR


def setup_logging() -> None:
    # Streamlit заново выполняет весь скрипт при каждом действии пользователя —
    # без этой проверки обработчики логов дублировались бы на каждый rerun
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
