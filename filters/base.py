import logging
from typing import Any, Callable, Dict, List

from pyspark.sql import DataFrame, SparkSession

from filters.constants import PrecomputedFeatureName
from utils import initialize_logger
from spark.constants import NUM_OUTPUT_PARTITIONS, SPARK_CACHE_DIR

FilterFunc = Callable[..., Any]
PrecomputedFeatures = Dict[PrecomputedFeatureName, DataFrame]

LOGGER: logging.Logger = initialize_logger()


class MetricFilterPipeline:
    def __init__(self):
        """
        Pipeline for applying filters to a dataset.
        """
        self.filters: List[FilterFunc] = []
        self.features: PrecomputedFeatures = {}
        self.spark: SparkSession

    def register_filter(self) -> FilterFunc:
        """
        Decorator for registering a filter function to the pipeline.

        Returns:
            FilterFunc: Decorated filter function
        """

        def decorator(filter_func: FilterFunc) -> FilterFunc:
            def wrapper(*args, **kwargs) -> Any:
                return filter_func(*args, **kwargs)

            LOGGER.info(f"Registering filter {filter_func.__name__}...")
            self.filters.append(filter_func)

            return wrapper

        return decorator

    def register_features(self, features: PrecomputedFeatures) -> None:
        """
        Register precomputed features to the pipeline.

        Args:
            features (PrecomputedFeatures): Precomputed features

        Returns:
            None
        """
        LOGGER.info(f"Registering features {features.keys()}...")
        self.features.update(features)

    def register_spark_session(self, spark: SparkSession) -> None:
        """
        Register Spark session to the pipeline.

        Args:
            spark (SparkSession): Spark session

        Returns:
            None
        """
        self.spark = spark

    def transform(self, original: DataFrame) -> DataFrame:
        """
        Apply all filters to the dataset.

        Args:
            original (DataFrame): Original dataset

        Returns:
            DataFrame: Filtered dataset
        """
        current_dataset = original

        for filter_func in self.filters:
            # Checkpointing each filter to side-step potential OOM issues
            LOGGER.info(f"Running filter {filter_func.__name__}...")
            current_dataset: DataFrame = filter_func(current_dataset, self.features).checkpoint()

        return current_dataset


PIPELINE_SINGLETON = MetricFilterPipeline()
