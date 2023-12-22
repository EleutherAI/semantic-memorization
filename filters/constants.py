from enum import Enum


class PrecomputedFeatureName(Enum):
    SEQUENCE_FREQUENCIES = "sequence_frequencies"
    MEMORIZED_TOKEN_FREQUENCIES = "memorized_token_frequencies"
    NON_MEMORIZED_TOKEN_FREQUENCIES = "non_memorized_token_frequencies"
    IS_CODE = "is_code"
    SEMANTIC_SNOWCLONES = "semantic_snowclones"
    SEMANTIC_TEMPLATES = "semantic_templates"
