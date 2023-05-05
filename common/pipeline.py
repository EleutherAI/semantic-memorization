from typing import Any, Callable, List, Set, Tuple, TypeVar

import pandas as pd

FilterFunc = TypeVar('FilterFunc', bound=Callable[..., Any])

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

    def run_pipeline(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for filter_func, output_column in self.filters:
            dataframe[output_column] = dataframe.apply(filter_func, axis=1)
        
        return dataframe
