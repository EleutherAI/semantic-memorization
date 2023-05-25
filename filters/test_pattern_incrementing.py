from .pattern_incrementing import incrementing_sequences_filter

def test_pattern_incrementing():
    text = "128. 129. 130. 131. 132. 133."
    assert incrementing_sequences_filter(text) == True

def test_pattern_new_lines_incrementing():
    text = "128.\n129.\n130.\n131.\n132.\n133."
    assert incrementing_sequences_filter(text) == True

def test_pattern_list_incrementing():
    text = "- 128.\n- 129.\n- 130.\n- 131.\n- 132.\n- 133."
    assert incrementing_sequences_filter(text) == True

def test_incrementing_nonnumerical_pattern():
    text = """![](edinbmedj75052-0047-b){#f5.123}

![](edinbmedj75052-0049-a){#f6.125}

![](edinbmedj75052-0049-b){#f7.125}

![](edin"""

    assert incrementing_sequences_filter(text) == True