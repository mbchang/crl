from collections import OrderedDict

class Math_to_Language(object):
    def __init__(self):
        self.vocabulary = None
        self.reverse_vocabulary = None

    def translate(self, math_expression):
        translation = []
        for e in math_expression:
            translation.append(self.vocabulary[e])
        return translation

class Math_to_English(Math_to_Language):
    def __init__(self):
        self.vocabulary = OrderedDict([
            ("0", "zero"),
            ("1", "one"),
            ("2", "two"),
            ("3", "three"),
            ("4", "four"),
            ("5", "five"),
            ("6", "six"),
            ("7", "seven"),
            ("8", "eight"),
            ("9", "nine"),
            ("+", "plus"),
            ("*", "times"),
            ("-", "minus"),
        ])
        self.reverse_vocabulary = {v: k for k,v in self.vocabulary.iteritems()}

class Math_to_Spanish(Math_to_Language):
    def __init__(self):
        self.vocabulary = OrderedDict([
            ("0", "cero"),
            ("1", "uno"),
            ("2", "dos"),
            ("3", "trs"),
            ("4", "cuatro"),
            ("5", "cinco"),
            ("6", "seis"),
            ("7", "siete"),
            ("8", "ocho"),
            ("9", "nueve"),
            ("+", "mas"),
            ("*", "por"),
            ("-", "menos"),
        ])
        self.reverse_vocabulary = {v: k for k,v in self.vocabulary.iteritems()}

class Language_to_PigLatin(object):
    def __init__(self):
        self.alphabet = [chr(letter) for letter in range(97,123)]
        self.vowels = ['a', 'e', 'i', 'o', 'u']
        self.consonants = [letter for letter in self.alphabet if letter not in self.vowels]

        self.vocabulary = None

    def translate(self, language_expression):
        translation = []
        for e in language_expression:
            if e[0] in self.vowels:
                translated = e + 'yay'
            else:
                # find the first vowel
                index_of_first_vowel = -1
                for i in range(len(e)):
                    if e[i] in self.vowels:
                        index_of_first_vowel = i
                        break
                translated = e[index_of_first_vowel:] + e[:index_of_first_vowel] + 'ay'
            translation.append(translated)
        return translation

class English_to_PigLatin(Language_to_PigLatin):
    def __init__(self):
        super(English_to_PigLatin, self).__init__()
        self.vocabulary = OrderedDict()
        for word in Math_to_English().vocabulary.values():
            self.vocabulary[word] = self.translate([word])[0]

class Spanish_to_PigSpanish(Language_to_PigLatin):
    def __init__(self):
        super(Spanish_to_PigSpanish, self).__init__()
        self.vocabulary = OrderedDict()
        for word in Math_to_Spanish().vocabulary.values():
            self.vocabulary[word] = self.translate([word])[0]

class Language_to_ReverseLanguage(object):
    def __init__(self):
        self.vocabulary = None

    def translate(self, language_expression):
        translation = []
        for e in language_expression:
            translation.append(e[::-1])
        return translation

class English_to_ReverseEnglish(Language_to_ReverseLanguage):
    def __init__(self):
        super(English_to_ReverseEnglish, self).__init__()
        self.vocabulary = OrderedDict()
        for word in Math_to_English().vocabulary.values():
            self.vocabulary[word] = self.translate([word])[0]

class Spanish_to_ReverseSpanish(Language_to_ReverseLanguage):
    def __init__(self):
        super(Spanish_to_ReverseSpanish, self).__init__()
        self.vocabulary = OrderedDict()
        for word in Math_to_Spanish().vocabulary.values():
            self.vocabulary[word] = self.translate([word])[0]