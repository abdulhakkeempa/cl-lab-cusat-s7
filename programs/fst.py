def pluralize_word_fst(word):
    state = 'START'
    result = word

    if word.endswith("^s#"):
        if word.endswith("x^s#"):
            state = 'ADD_ES'
        elif word.endswith("s^s#"):
            state = 'ADD_ES'
        elif word.endswith("z^s#"):
            state = 'ADD_ES'
        else:
            state = 'ADD_S'
    else:
        return word

    if state == 'ADD_ES':
        result = word.replace("^s#", "es")
    elif state == 'ADD_S':
        result = word.replace("^s#", "s")

    return result

test_cases = ["fox^s#", "boy^s#", "bus^s#", "quiz^s#", "dog^s#"]
results = {word: pluralize_word_fst(word) for word in test_cases}
print(results)
