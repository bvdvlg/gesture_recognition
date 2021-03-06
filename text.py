from nltk.corpus import words
from nltk import download
download('words')
from difflib import get_close_matches
from image_container import Constants


class Text:
    def __init__(self, string=""):
        self.text = string
        self.alphabet = {}
        self.reversed_alphabet = {}
        self.current_word = ""
        self.english_dictionary = words.words()
        self.use_t9 = True

    def set_alphabets(self, alphabet):
        self.alphabet = alphabet
        self.reversed_alphabet = {el: num for num, el in self.alphabet.items()}

    def get_sym(self, num):
        return self.reversed_alphabet[num]

    def get_num(self, num):
        return self.reversed_alphabet[num]

    def append(self, sym):
        if sym != " ":
            self.current_word += sym
            return self

        last_word = list()
        if self.use_t9:
            last_word = get_close_matches(self.current_word, self.english_dictionary, n=1, cutoff=Constants.Thresholds.CUTOFF_TRESHOLD)

        if last_word:
            self.text += last_word[0]
        else:
            self.text += self.current_word

        self.current_word = ""
        self.text += " "
        return self
