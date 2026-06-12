"""logger — 统一日志 ==============================================
同时输出到控制台与 logs/ 文件,方便事后排查。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


_LOG_DIR: Path = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取带文件和控制台输出的 logger 实例。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复绑定

    logger.setLevel(level)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件(按大小轮转,最大 2MB × 3 个)
    fh = RotatingFileHandler(
        _LOG_DIR / "app.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger