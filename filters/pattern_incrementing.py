import re

def incrementing_sequences_filter(text: str) -> bool:
    """
    This sequence will classify a given text is an incrementing sequence or not.

    Args:
        text (str): The current sequence to be classified.

    Returns:
        bool: Whether the sequence is an incrementing sequence or not.
    """
    # Split by seperators between text
    possible_seperators = list(set(re.findall(r'(?<=\d)(\D+)(?=\d)', text))) + [" "] + ["\n"]
    for seperator in possible_seperators:
    # seperator = ""
    # reading = None
    # prev_char = None
    # for index, character in enumerate(text):
    #     next_char = text[index + 1] if index + 1 < len(text) else ""
    #     if prev_char is None:
    #         prev_char = character
    #     if not character.isdigit() and not next_char.isdigit():
    #         reading = True
    #         seperator += character
    #     if character.isdigit() and reading is True:
    #         break

    #     prev_char = character
        split_text = text.split(" " if seperator == "" else seperator)

    # trim the end if the final character(s) is a seperator
        trailing_seperator = ""
        for sep_index in range(len(seperator)):
            if text.split(seperator)[-1][sep_index - 1:] == seperator[:sep_index + 1]:
                trailing_seperator += seperator[:sep_index + 1]
            else:
                break
        split_text[-1] = split_text[-1][:-len(trailing_seperator)]

        # Check if the sequence is just a list of digits
        if len(split_text) == 1:
            failed = False
            prev_char = None
            is_decrementing = None
            for char in split_text[0]:
                if char.isdigit():
                    if prev_char is None and is_decrementing is None:
                        prev_char = char
                    elif is_decrementing is None:
                        is_decrementing = int(char) < int(prev_char)
                        prev_char = char
                    elif is_decrementing and (int(char) < int(prev_char)):
                        prev_char = char
                    elif not is_decrementing and (int(char) > int(prev_char)):
                        prev_char = char
                    else:
                        failed = True
                        break
                else:
                    failed = True
                    break
            if failed:
                return False


    return True