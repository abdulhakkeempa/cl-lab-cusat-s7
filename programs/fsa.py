def is_plural_noun_accepted_fsa(word):
    if len(word) < 2 or word[-1] != 's':
        return False

    word = word[::-1]
    state = 'S1'

    for char in word[1:]:
        if state == 'S1':
            if char == 'y':
                state = 'S2'
            elif char == 'e':
                state = 'S3'
            else:
                return False
        elif state == 'S2':
            if char in 'aeiou':
                state = 'S5'
            else:
                return False
        elif state == 'S3':
            if char == 'i':
                state = 'S4'
            else:
                return False
        elif state == 'S4':
            if char.isalpha() and char not in 'aeiou':
                state = 'S6'
            else:
                return False
        elif state == 'S5':
            continue
        elif state == 'S6':
            continue

    return True

test_words = ['boys', 'toys', 'ponies', 'skies', 'puppies', 'boies', 'toies', 'ponys', 'carries', 'daisies']
results = {word: is_plural_noun_accepted_fsa(word) for word in test_words}

print(results)
