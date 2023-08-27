import os
from typing import Any, Callable, List, Tuple, TypeVar

import pandas as pd
from pandarallel import pandarallel

FilterFunc = TypeVar("FilterFunc", bound=Callable[..., Any])
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())


class MetricFilterPipeline:
    def __init__(self):
        self.filters: List[Tuple[FilterFunc, str]] = []

    def register_filter(self, output_column: str) -> FilterFunc:
        def decorator(filter_func: FilterFunc) -> FilterFunc:
            def wrapper(*args, **kwargs) -> Any:
                return filter_func(*args, **kwargs)

            self.filters.append((filter_func, output_column))

            return wrapper

        return decorator

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for filter_func, output_column in self.filters:
            dataframe[output_column] = dataframe.parallel_apply(filter_func, axis=1)

        return dataframe


PIPELINE_SINGLETON = MetricFilterPipeline()
