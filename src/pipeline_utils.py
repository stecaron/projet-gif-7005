from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from word_to_remove_factory import WordsToRemoveFactory


class CurrentModel(BaseEstimator):
    def __init__(self, model_name):
        self.model_name = model_name

    def fit(self, x, y=None):
        self.update_current_model()

        return self.current_model_class.fit(x, y=y)

    def predict(self, x):
        return self.current_model_class.predict(x)

    def update_current_model(self):
        if self.model_name == "knn":
            self.current_model_class = KNeighborsClassifier()

        else:
            raise TypeError("{} is not a selectable model".format(self.model_name))

    def score(self, x, y=None):
        return self.current_model_class.score(x, y=y)



class AddOtherFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_add):
        self.feature_to_add = feature_to_add

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if self.feature_to_add == "old_query":
            raise NotImplementedError()
        else:
            return None



class RemoveWords(BaseEstimator):
    def __init__(self, words_to_remove, words_to_remove_factory=None):
        self.word_remover = None
        self.words_to_remove = words_to_remove
        self.current_words_to_remove_factory = words_to_remove_factory

        if self.current_words_to_remove_factory is None:
            self.current_words_to_remove_factory = WordsToRemoveFactory()

        super().__init__()

    def fit(self, x, y=None):
        self.word_remover = self.current_words_to_remove_factory.create_word_to_remove_function(self.words_to_remove)

        return self

    def transform(self, x):
        return self.word_remover.transform(x=x)



class NormaliseWords(BaseEstimator, TransformerMixin):
    def __init__(self, normalise_type):
        self.normalise_type = normalise_type

        if self.normalise_type == "lemmatize" or "stemming" or "none":
            pass
        else:
            raise TypeError("{} is not lemmatize, stemming or none".format(self.normalise_type))

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if self.normalise_type == "none":
            x = np.array(list(x))
        elif self.normalise_type == "lemmatize":
            x = map(lambda r: ' '.join([wordnet.WordNetLemmatizer().lemmatize(i.lower()) for i in r.split()]), x)
            x = np.array(list(x))
        elif self.normalise_type == "stemming":
            x = map(lambda r: ' '.join([porter.PorterStemmer().stem(i.lower()) for i in r.split()]), x)
            x = np.array(list(x))
        return x