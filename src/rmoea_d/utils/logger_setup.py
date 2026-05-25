"""
Logging setup: console for key info, file for detailed logs.
日志配置模块：关键信息输出到控制台，详细日志写入文件。
"""

import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Setup logging with both console and file handlers.
    配置日志系统：控制台输出关键信息，文件记录详细日志。
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rmoea_d_{timestamp}.log")

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # Console handler - only INFO and above, concise format
    # 控制台处理器：仅输出INFO及以上级别，简洁格式
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - DEBUG and above, detailed format with timestamps
    # 文件处理器：记录DEBUG及以上级别，包含时间戳的详细格式
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    logging.info("Logging initialized. Console: INFO+, File: DEBUG+ -> %s", log_file)
    return log_file
