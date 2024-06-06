import findspark
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from spark.constants import *


def initialize_spark() -> SparkSession:
    """
    Initializes the Spark session.

    Returns:
        SparkSession: Spark session.
    """
    findspark.init()

    config = (
        SparkConf()
        .setMaster(f"local[{NUM_CPU_COUNT}]")
        .setAppName("semantic-memorization")
        # .set("spark.driver.cores", f"12")
        .set("spark.driver.memory", f"{DRIVER_MEMORY_SIZE}")
        .set("spark.driver.memoryOverhead", f"{HEAP_SIZE}")
        .set("spark.ui.port", "5050")
        # .set("spark.executor.cores", f"{NUM_CPU_COUNT - 16}")
        .set("spark.sql.shuffle.partitions", f"{NUM_SPARK_PARTITIONS}")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "100000")
        .set("spark.memory.offHeap.enabled", "true")
        .set("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
        .set("spark.memory.offHeap.size", f"{HEAP_SIZE}")
        .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
    )
    session = SparkSession.builder.config(conf=config).getOrCreate()
    session.sparkContext.setCheckpointDir(SPARK_CACHE_DIR)

    return session
