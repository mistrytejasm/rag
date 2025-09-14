import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        # Fix Unicode encoding for Windows
        formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
