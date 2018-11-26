import nltk
from nltk import pos_tag, word_tokenize


class WordsToRemoveFactory:

    def create_word_to_remove_function(self, words_to_remove):
        word_remover = None

        if words_to_remove == "none":
            word_remover = WordRemover()
        elif words_to_remove == "tool_words":
            word_remover = ToolWords(nltk.corpus.stopwords.words('english'))
        elif words_to_remove == "closed_class":
            word_remover = ClosedClass()
        elif words_to_remove == "tool_words_and_closed_class":
            word_remover = ToolWordsAndClosedClass(nltk.corpus.stopwords.words('english'))

        if word_remover is None:
            raise TypeError("{} is not a removal pattern".format())

        return word_remover


class WordRemover:

    def transform(self, x):
        return x

    def fit(self, _):
        return self


class ToolWords(WordRemover):

    def __init__(self, stopwords):
        self.stopwords = stopwords

    def transform(self, x):
        return map(lambda r: ' '.join([word for word in r.split() if word.lower() not in self.stopwords]), x)


class ClosedClass(WordRemover):

    def __init__(self, tag_set='universal', open_class=['ADJ', 'ADV', 'NOUN', 'VERB'], pos_tagger=pos_tag, word_tokeniser=word_tokenize):
        self.tag_set = tag_set
        self.open_class = open_class
        self.pos_tag = pos_tagger
        self.word_tokeniser = word_tokeniser

    def transform(self, x):
        return map(lambda r: ' '.join([word for word, tag in self.pos_tag(self.word_tokeniser(r), tagset=self.tag_set) if tag in self.open_class]), x)


class ToolWordsAndClosedClass(WordRemover):

    def __init__(self, stopwords, tag_set='universal', open_class=['ADJ', 'ADV', 'NOUN', 'VERB'], pos_tagger=pos_tag,
                 word_tokeniser=word_tokenize):
        self.stopwords = stopwords
        self.tag_set = tag_set
        self.open_class = open_class
        self.pos_tag = pos_tagger
        self.word_tokeniser = word_tokeniser

    def transform(self, x):
        return map(lambda r: ' '.join(
            [word for word, tag in self.pos_tag(self.word_tokeniser(r), tagset=self.tag_set) if
             tag in self.open_class]), x)


