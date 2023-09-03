import logging
from typing import Any, Callable, Dict, List, TypeAlias

from pyspark.sql import DataFrame

from filters.constants import PrecomputedFeatureName
from utils import initialize_logger

FilterFunc: TypeAlias = Callable[..., Any]
PrecomputedFeatures: TypeAlias = Dict[PrecomputedFeatureName, DataFrame]

LOGGER: logging.Logger = initialize_logger()


class MetricFilterPipeline:
    def __init__(self):
        self.filters: List[FilterFunc] = []
        self.features: PrecomputedFeatures = {}

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

    def transform(self, dataset: DataFrame) -> DataFrame:
        for filter_func in self.filters:
            dataset = filter_func(dataset, self.features)

        return dataset


PIPELINE_SINGLETON = MetricFilterPipeline()
