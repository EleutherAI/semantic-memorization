import logging
import sys

import findspark
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession


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


def initialize_spark() -> SparkSession:
    """
    Initializes the Spark session.

    Returns:
        SparkSession: Spark session.
    """
    findspark.init()

    config = (
        SparkConf()
        .setMaster("local[*]")
        .setAppName("semantic-memorization")
        .set("spark.driver.cores", "128")
        .set("spark.driver.memory", "128g")
        .set("spark.driver.memoryOverheadFactor", "0.2")
    )

    return SparkSession.builder.config(conf=config).getOrCreate()
