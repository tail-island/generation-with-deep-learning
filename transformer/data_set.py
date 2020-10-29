import numpy as np

from functools import reduce
from funcy     import *
from operator  import add, sub, mul


NUMBER_OF_DIGITS = 3
WORDS            = tuple(concat((' ',), ('+', '-', '*', '/'), map(str, range(10)), ('^', '$')))
ENCODE           = dict(zip(WORDS, count()))
DECODE           = dict(zip(count(), WORDS))


def create_sentences_collection():
    def create_number():
        return reduce(lambda acc, x: acc * 10 + x, take(np.random.randint(1, NUMBER_OF_DIGITS + 1), np.random.randint(0, 10, (NUMBER_OF_DIGITS + 1,))))

    def create_sentence(word, op):
        x = create_number()
        y = create_number()
        z = op(x, y)

        return str(x) + word + str(y), str(z)

    return zip(*concat(repeatedly(partial(create_sentence, '+', add), 10000),
                       repeatedly(partial(create_sentence, '-', sub), 10000),
                       repeatedly(partial(create_sentence, '*', mul), 10000)))


def encode(sentence, max_sentence_length):
    return take(max_sentence_length + 2, concat((ENCODE['^'],),
                                                map(ENCODE, sentence),
                                                (ENCODE['$'],),
                                                (ENCODE[' '],) * max_sentence_length))


def decode(encoded):
    return ''.join(map(DECODE, encoded))


def create_dataset():
    x_strings, y_strings = create_sentences_collection()

    x_string_max_length = max(map(len, x_strings))
    y_string_max_length = max(map(len, y_strings))

    temp_xs = np.array(tuple(map(lambda x_string: tuple(encode(x_string, x_string_max_length)), x_strings)))
    temp_ys = np.array(tuple(map(lambda y_string: tuple(encode(y_string, y_string_max_length)), y_strings)))

    indexes = np.random.permutation(np.arange(len(temp_xs)))
    xs      = temp_xs[indexes]
    ys      = temp_ys[indexes]

    return (xs[:-100], ys[:-100]), (xs[-100:], ys[-100:])
