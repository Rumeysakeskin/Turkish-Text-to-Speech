import re
import math

single_digit_conv_dict = {
    0: "sıfır",
    1: "bir",
    2: "iki",
    3: "üç",
    4: "dört",
    5: "beş",
    6: "altı",
    7: "yedi",
    8: "sekiz",
    9: "dokuz"
}

double_digit_conv_dict = {
    1: "on",
    2: "yirmi",
    3: "otuz",
    4: "kırk",
    5: 'elli',
    6: 'atmış',
    7: 'yetmiş',
    8: 'seksen',
    9: 'doksan'
}

factorizations_list = [
    [1e2, "yüz"],
    [1e3, "bin"],
    [1e6, "milyon"],
    [1e9, "milyar"],
    [1e12, "trilyon"],
    [1e15, "katrilyon"],
    [1e18, "kentilyon"]
]


def convert_integer_pronunciation(val: int) -> str:
    """
    Converts given integer to its turkish text pronunciation
    :param val: Integer number
    :return: Turkish text representation of integer number
    """
    if val < 10:
        return single_digit_conv_dict[val]

    elif val < 100:
        pronunciation = double_digit_conv_dict[math.floor(val / 10)]
        if val % 10 == 0:
            return pronunciation
        return double_digit_conv_dict[math.floor(val / 10)] + " " + convert_integer_pronunciation(val % 10)

    else:
        for idx, (divider, factorization_name) in enumerate(factorizations_list):
            next_factorization_divider = factorizations_list[idx + 1][0]

            if val < next_factorization_divider:
                first_part = math.floor(val / divider)
                second_part = val % divider

                if first_part != 1:
                    pronunciation = convert_integer_pronunciation(first_part) + " "
                else:
                    pronunciation = ""

                pronunciation = pronunciation + factorization_name

                if second_part != 0:
                    pronunciation += " " + convert_integer_pronunciation(second_part)

                return pronunciation


def find_and_normalize_number(text: str) -> str:

    result = re.search("[0-9]+", text)
    found = result is not None

    if found:
        value = int(result.group())
        pronunciation = convert_integer_pronunciation(value)
        text = list(text)
        text[result.start():result.end()] = pronunciation
        text = "".join(text)

    return found, text


def normalize_numbers(text: str) -> str:
    # Iteratively search and replace number pronunciation, this is basically emulation of do-while on python
    number_normalized, new_text = find_and_normalize_number(text)
    while number_normalized:
        number_normalized, new_text = find_and_normalize_number(new_text)

    return new_text


def find_filter_replace(text: str, find_regex: str, replace_search: str, replacement_word: str) -> str:
    """

    :param text: Text
    :param find_regex: Regex that is used to find parts that replacement operations will run on
    :param replace_search: Regex that will specifically match to the part that we want to replace
    :param replacement_word: Text that we want to replace the part that is matched by replace_search
    :return: Text with indicated parts replaced with replacement_word
    """
    def find_filter_replace_(text, find_regex, replace_search, replacement_word):
        match = re.search(find_regex, text)
        if match is None:
            return False, text

        matched_text = match.group()
        replace_text = matched_text.replace(replace_search, replacement_word)

        text = list(text)
        text[match.start():match.end()] = replace_text
        text = "".join(text)
        return True, text

    success, text = find_filter_replace_(text, find_regex, replace_search, replacement_word)
    while success:
        success, text = find_filter_replace_(text, find_regex, replace_search, replacement_word)

    return text


def normalize_punctuations(text: str) -> str:

    text = find_filter_replace(text, "[1-9]\.[0-9][. °-]", ".5", " buçuk")
    text = find_filter_replace(text, "[1-9]\,[0-9][. °-]", ",5", " buçuk")
    text = find_filter_replace(text, "[0-9]\.[0-9]", ".", " nokta ")

    text = find_filter_replace(text, "[0-9]% ", "%", "")
    text = find_filter_replace(text, " %[0-9]", "%", "yüzde ")

    text = text.replace("-", " ")
    text = text.replace("°", " derece")
    text = text.replace("+", " artı ")
    text = text.replace("/", " bölü ")
    text = text.replace("*", " çarpı ")

    text = re.sub("[()]", " ", text)

    text = text.replace(" gr ", " gram ")
    text = text.replace(" dk ", " dakika ")
    text = text.replace(" ml ", " mililitre ")
    return text


def normalize_text(text: str) -> str:
    """
    Normalizes punctuations and numbers to their turkish text representation
    """
    # Writes numbers in text format, converts punctuations based on their usage in sentence, eliminate extra spaces
    text = normalize_punctuations(text)
    text = normalize_numbers(text)
    text = re.sub(" +", " ", text).strip()
    return text


# def conduct_tests():
#     """
#     Runs tests over system to ensure if everything is working correctly and help locate any bugs
#     :return: None
#     """
#
#     # First elements are inputs and second ones are expected outputs
#     sentences = [
#         ["Fırını ısıtmadan önce 1.5 gr tuz atınız.",
#          "Fırını ısıtmadan önce bir buçuk gram tuz atınız."],
#
#         ["1/3-1/4 dk boyunca pişirip %100 kızarmasını bekleyin.",
#          "bir bölü üç bir bölü dört dakika boyunca pişirip yüzde yüz kızarmasını bekleyin."],
#
#         ["ben 1.25 ve 1394.47 gibi karışık rakamları düzgün söyleyebilirim.",
#          "ben bir nokta yirmi beş ve bin üç yüz doksan dört nokta kırk yedi gibi karışık rakamları düzgün söyleyebilirim."],
#
#         ["568415193941344131",
#          "beş yüz atmış sekiz katrilyon dört yüz on beş trilyon yüz doksan üç milyar dokuz yüz kırk bir milyon "
#          "üç yüz kırk dört bin yüz yirmi sekiz"],
#
#         ["123 gram şeker (535 ml su bardağı)",
#          "yüz yirmi üç gram şeker beş yüz otuz beş mililitre su bardağı"],
#
#         ["Kekin sosu için kakao, şeker, nişasta, vanilin ve sütü sos tenceresine alıp orta ateşte kıvam alıncaya kadar karıştırın.",
#          "Kekin sosu için kakao, şeker, nişasta, vanilin ve sütü sos tenceresine alıp orta ateşte kıvam alıncaya kadar karıştırın."],
#
#         ["Uzun  boşluk      kaldırma      testi",
#          "Uzun boşluk kaldırma testi"],
#
#         ["1.5-2.25 gr tuz atınız.",
#          "bir buçuk iki nokta yirmi beş gram tuz atınız."]
#     ]
#
#     results = np.zeros(len(sentences), dtype=np.uint8)
#     for idx, (stn_in, stn_out) in enumerate(sentences):
#         cleaned_out = normalize_text(stn_in)
#         if cleaned_out == stn_out:
#             results[idx] = 1
#         else:
#             print("- " * 15)
#             print("Test failed at idx", idx)
#             print(stn_in)
#             print("[" + stn_out + "]")
#             print("[" + cleaned_out + "]")
#
#     print("- " * 30)
#     print(np.sum(results), "out of", len(results), "passed")
#
#
# if __name__ == '__main__':
#     conduct_tests()
