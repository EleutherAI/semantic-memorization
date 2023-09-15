from .pattern_incrementing import incrementing_sequences_filter_wrapper


def test_pattern_incrementing_no_space():
    text = "123456789"
    assert incrementing_sequences_filter_wrapper(text) == True


def test_pattern_incrementing_no_space_with_char():
    text = "1A23456789"
    assert incrementing_sequences_filter_wrapper(text) == False


def test_pattern_incrementing():
    text = "12.8. 12.9. 13.0. 13.1. 13.2. 13.3."
    assert incrementing_sequences_filter_wrapper(text) == True


def test_pattern_new_lines_incrementing():
    text = "128.\n129.\n130.\n131.\n132.\n133."
    assert incrementing_sequences_filter_wrapper(text) == True


def test_pattern_list_incrementing():
    text = "- 128.\n- 129.\n- 130.\n- 131.\n- 132.\n- 133."
    assert incrementing_sequences_filter_wrapper(text) == True


def test_incrementing_nonnumerical_pattern():
    text = """
![](edinbmedj75052-0047-b){#f5.123}
![](edinbmedj75052-0049-a){#f6.125}
![](edinbmedj75052-0049-b){#f7.125}
![](edin
"""
    assert incrementing_sequences_filter_wrapper(text) == True


def test_incrementing_seminnumerical_pattern():
    text = "A.1 , A.2 , A.3 , A.4, B.1 , B.2, B.3, C.1"
    assert incrementing_sequences_filter_wrapper(text) == True
