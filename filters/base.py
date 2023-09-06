import logging
from typing import Any, Callable, Dict, List, TypeAlias

from pyspark.sql import DataFrame, SparkSession

from filters.constants import PrecomputedFeatureName
from utils import initialize_logger
from spark.constants import NUM_PARTITIONS, SPARK_CACHE_DIR

FilterFunc: TypeAlias = Callable[..., Any]
PrecomputedFeatures: TypeAlias = Dict[PrecomputedFeatureName, DataFrame]

LOGGER: logging.Logger = initialize_logger()


class MetricFilterPipeline:
    def __init__(self):
        self.filters: List[FilterFunc] = []
        self.features: PrecomputedFeatures = {}
        self.spark: SparkSession

    def register_filter(self) -> FilterFunc:
        def decorator(filter_func: FilterFunc) -> FilterFunc:
            def wrapper(*args, **kwargs) -> Any:
                return filter_func(*args, **kwargs)

            LOGGER.info(f"Registering filter {filter_func.__name__}...")
            self.filters.append(filter_func)

            return wrapper

        return decorator

    def register_features(self, features: PrecomputedFeatures) -> None:
        LOGGER.info(f"Registering features {features.keys()}...")
        self.features.update(features)

    def register_spark_session(self, spark: SparkSession) -> None:
        self.spark = spark

    def transform(self, original: DataFrame) -> DataFrame:
        current_dataset = original

        for filter_func in self.filters:
            # Checkpointing each filter to side-step potential OOM issues
            current_dataset: DataFrame = filter_func(current_dataset, self.features).checkpoint()

        return current_dataset


PIPELINE_SINGLETON = MetricFilterPipeline()
