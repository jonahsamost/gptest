import os
import logging

def setup_logging():
    logger = logging.getLogger('GPTest')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler) 
    logger.propagate = False
    return logger

logger = setup_logging()

