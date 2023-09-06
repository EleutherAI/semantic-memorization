import logging
import sys


def initialize_formatter() -> logging.Formatter:
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z")


def initialize_logger() -> logging.Logger:
    logger = logging.getLogger("semantic-memorization")

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(initialize_formatter())
        logger.addHandler(stdout_handler)

    return logger
