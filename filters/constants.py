from enum import Enum


class PrecomputedFeatureName(Enum):
    SEQUENCE_FREQUENCIES = "sequence_frequencies"
    MEMORIZED_TOKEN_FREQUENCIES = "memorized_token_frequencies"
    NON_MEMORIZED_TOKEN_FREQUENCIES = "non_memorized_token_frequencies"
