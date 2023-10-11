import findspark
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from spark.constants import NUM_CPU_COUNT, NUM_SPARK_PARTITIONS, SPARK_CACHE_DIR


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
        .set("spark.driver.cores", f"{NUM_CPU_COUNT}")
        .set("spark.driver.memory", "80g")
        .set("spark.driver.memoryOverhead", "16g")
        .set("spark.sql.shuffle.partitions", f"{NUM_SPARK_PARTITIONS}")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .set("spark.memory.offHeap.enabled", "true")
        .set("spark.memory.offHeap.size", "16g")
        .set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
    )
    session = SparkSession.builder.config(conf=config).getOrCreate()
    session.sparkContext.setCheckpointDir(SPARK_CACHE_DIR)

    return session
