from .base import PIPELINE_SINGLETON as PIPELINE
# The import here determines the order of the pipeline
from .highly_duplicated_filter import token_frequency_filter
from .highly_duplicated_filter import sequence_frequency_filter
