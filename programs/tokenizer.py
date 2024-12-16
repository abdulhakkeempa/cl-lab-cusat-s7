import re

def segment_text(input_text):
    token_pattern = r"""
    (?:[A-Za-z]\.){2,}[A-Z]
    | \w+(?:-\w+)+
    | \b([A-Za-z]+)(n't|'s|'ll|'em|'ve|'re|'d)\b
    | \b\w+\b
    | [.,!?;"()\[\]{}<>]
    """

    segments = []  # List to store tokens

    for match in re.finditer(token_pattern, input_text, flags=re.VERBOSE):
        if match.group(1):
            segments.extend([match.group(1), match.group(2)])
        else:
            segments.append(match.group(0))

    return segments

text = "isn't"
unique_segments = set(segment_text(text))
print(unique_segments)
