from .base import PIPELINE_SINGLETON as PIPELINE
# The import here determines the order of the pipeline
from .detokenize import detokenize
from .highly_duplicated_filter import sequence_duplicates_filter
from .token_frequency_statistics_filter import token_frequency_statistics_filter
from .pattern_incrementing import incrementing_sequences_filter
from .highly_repetitive import highly_repetitive_filter
