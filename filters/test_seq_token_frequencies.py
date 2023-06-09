from .seq_token_frequencies import seq_token_frequencies

def test_seq_token_frequencies():
    """Test seq_token_frequencies function."""
    tokens = [1233, 1233, 1234, 1235, 1235, 1235, 1236, 1236, 1236, 1236]
    token_frequencies = {1233: 40, 1234: 800, 1235: 390, 1236: 4}
    assert seq_token_frequencies(tokens, token_frequencies) == {1233: 40, 1234: 800, 1235: 390, 1236: 4}