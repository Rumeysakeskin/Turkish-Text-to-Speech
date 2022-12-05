from typing import List

from turkish_text_normalizer import normalize_text
import re

_pad        = '_'
_punctuation = '!\'(),.:;?\" '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÖÇIĞÜŞabcdefghijklmnopqrstuvwxyzöçığüş'
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_whitespace_re = re.compile(r'\s+')


def _intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'


def _add_blank(text_list):
    text = _intersperse(text_list, len(symbols))
    return text


def collapse_whitespace(text: str):
    return re.sub(_whitespace_re, ' ', text)


def _text_to_sequence(text, add_blank=True):
    text = text.strip()
    sequence = _symbols_to_sequence(text)
    return sequence


def tokenize_text(text: str, add_blank: bool = True) -> List[int]:
    """
    Takes a text as input and converts it to list of characters where each character is represented by an integer
    token - it is of paramount importance that exact same tokenization mechanism is used for both training and inference
    :param text: Text to be tokenized
    :param add_blank: Whether to insert blanks between each character or not
    :return: Tokenized form of the text - which is a list of integers
    """
    text = text.lower()
    normalized_text = normalize_text(text)
    normalized_text = collapse_whitespace(normalized_text)

    if add_blank:
        tokens = _text_to_sequence(normalized_text.strip())
        tokens = _add_blank(tokens)
    else:
        normalized_text = " " + normalized_text.strip() + " "
        tokens = _text_to_sequence(normalized_text)

    return tokens


if __name__ == '__main__':
    ip = "deneme"
    print(tokenize_text(ip, True))
