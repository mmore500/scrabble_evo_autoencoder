from string import ascii_lowercase

VALID_CHARS = ascii_lowercase + ' '

CHAR_FREQ = {
        'a' : 9/120,
        'b' : 2/120,
        'c' : 2/120,
        'd' : 4/120,
        'e' : 12/120,
        'f' : 2/120,
        'g' : 3/120,
        'h' : 2/120,
        'i' : 9/120,
        'j' : 1/120,
        'k' : 1/120,
        'l' : 4/120,
        'm' : 2/120,
        'n' : 6/120,
        'o' : 8/120,
        'p' : 2/120,
        'q' : 1/120,
        'r' : 6/120,
        's' : 4/120,
        't' : 6/120,
        'u' : 4/120,
        'v' : 2/120,
        'w' : 2/120,
        'x' : 1/120,
        'y' : 2/120,
        'z' : 1/120,
        ' ' : 22/120
    }


PS = [CHAR_FREQ[x] for x in VALID_CHARS]

CHAR_IDX = {c : i for (i, c) in enumerate(VALID_CHARS)}

SCRABBLE_CHARACTER_SCORE = {
        'a' : 1,
        'b' : 3,
        'c' : 3,
        'd' : 2,
        'e' : 1,
        'f' : 4,
        'g' : 2,
        'h' : 4,
        'i' : 1,
        'j' : 8,
        'k' : 5,
        'l' : 1,
        'm' : 3,
        'n' : 1,
        'o' : 1,
        'p' : 3,
        'q' : 10,
        'r' : 1,
        's' : 1,
        't' : 1,
        'u' : 1,
        'v' : 4,
        'w' : 4,
        'x' : 8,
        'y' : 4,
        'z' : 10,
        ' ' : 0
    }

LETTER_CHARACTER_SCORE = {
        'a' : 1,
        'b' : 1,
        'c' : 1,
        'd' : 1,
        'e' : 1,
        'f' : 1,
        'g' : 1,
        'h' : 1,
        'i' : 1,
        'j' : 1,
        'k' : 1,
        'l' : 1,
        'm' : 1,
        'n' : 1,
        'o' : 1,
        'p' : 1,
        'q' : 1,
        'r' : 1,
        's' : 1,
        't' : 1,
        'u' : 1,
        'v' : 1,
        'w' : 1,
        'x' : 1,
        'y' : 1,
        'z' : 1,
        ' ' : 0
    }

def id(x):
    return x

STDPARAM = {
        'indpb' : 0.1,
        'mutpb' : 0.2,
        'npop' : 50,
        'ngen' : 500,
        'indsize' : 100,
        'nhof' : 1,
        'cscore' : LETTER_CHARACTER_SCORE,
        'gpmap' : id
    }
