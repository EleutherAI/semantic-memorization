from .base import PIPELINE_SINGLETON as PIPELINE

_has_registered_all_filters = False

if not _has_registered_all_filters:
    # The import here determines the order of the pipeline
    from .detokenize import detokenize
    from .pattern import pattern_sequences_filter
    from .highly_duplicated_filter import sequence_duplicates_filter
    from .token_frequency_statistics_filter import token_frequency_statistics_filter
    from .highly_repetitive import highly_repetitive_filter
    from .code_vs_nl import code_vs_nl_filter
    from .semantic_duplicates_filter import semantic_duplicates_filter
    from .huffman_coding_filter import huffman_coding_filter

    _has_registered_all_filters = True
