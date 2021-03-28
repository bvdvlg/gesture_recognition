from nltk.corpus import words
from difflib import get_close_matches

cutoff_threshold = 0.8


class Text():
    def __init__(self, string=""):
        self.text = string
        self.alphabet = {}
        self.reversed_alphabet = {}
        self.current_word = ""
        self.english_dictionary = words.words()

    def set_alphabets(self, alphabet):
        self.alphabet = alphabet
        self.alphabet[" "] = self.alphabet["space"]
        del self.alphabet["space"]
        self.reversed_alphabet = {el: num for num, el in self.alphabet.items()}

    def get_sym(self, num):
        return self.reversed_alphabet[num]

    def get_num(self, num):
        return self.reversed_alphabet[num]

    def append(self, sym):
        if sym != " ":
            self.current_word += sym
            return self
        last_word = get_close_matches(self.current_word, self.english_dictionary, n=1, cutoff=cutoff_threshold)

        if last_word:
            self.text += last_word[0]
        else:
            self.text += self.current_word

        self.current_word = ""
        self.text += " "
        return self
