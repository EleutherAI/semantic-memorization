from .base import PIPELINE_SINGLETON as PIPELINE

_has_registered_all_filters = False

if not _has_registered_all_filters:
    # The import here determines the order of the pipeline
    from .detokenize import detokenize
    from .highly_duplicated_filter import sequence_duplicates_filter
    from .token_frequency_statistics_filter import token_frequency_statistics_filter
    from .pattern_incrementing import incrementing_sequences_filter
    from .highly_repetitive import highly_repetitive_filter
    from .code_vs_nl import code_vs_nl_filter
    from .semantic_duplicates_filter import semantic_duplicates_filter

    _has_registered_all_filters = True
