def incrementing_sequences_filter(text: str) -> bool:
    """
    This sequence will classify a given text is an incrementing sequence or not.

    Args:
        text (str): The current sequence to be classified.

    Returns:
        bool: Whether the sequence is an incrementing sequence or not.
    """
    # Check for incrementing sequences of only numbers
    previous_entry = None
    direction = None
    for string_entry in text.split():
        string_entry = (string_entry[:-1] if string_entry[-1] == "." else string_entry)
        if string_entry.isdigit():
            numerical_entry = float(string_entry)
            if previous_entry is None:
                previous_entry = numerical_entry
                continue
            elif direction is None:
                direction = "positive" if numerical_entry > previous_entry else "negative"
                previous_entry = numerical_entry
                continue
            else:
                if direction == "positive":
                    if numerical_entry > previous_entry:
                        previous_entry = numerical_entry
                        continue
                    else:
                        return False
                else:
                    if numerical_entry < previous_entry:
                        previous_entry = numerical_entry
                        continue
                    else:
                        return False

        # A non-numerican entry was found
        return False

    return True