from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
#from word_to_remove_factory import WordsToRemoveFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np


class FilterColumns(BaseEstimator, TransformerMixin):
    """
    filter_group : liste des noms de columns du data frame qu'on veut conserver
    """
    def __init__(self, filter_group):
        self.filter_group = filter_group

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X[self.filter_group]

        return X


def RemoveStopWords(X, transformation_target="document_title"):
    stop_words = set(stopwords.words('english'))
    filtered_expression = []
    for i, sentence in enumerate(X[transformation_target]):
        words = word_tokenize(sentence)
        filtered_words = [w for w in words if not w in stop_words]
        filtered_expression.append(" ".join(word for word in filtered_words))
    X.loc[:, transformation_target] = filtered_expression
    return X

class NormalizeQuery(BaseEstimator, TransformerMixin):
    def __init__(self, normalize_method, transformation_target="query_expression"):
        self.normalize_method = normalize_method
        self.transformation_target = transformation_target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.normalize_method == "None":
            pass

        elif self.normalize_method == "PorterStemmer":
            ps = PorterStemmer()
            stemmed_queries = []
            for i, sentence in enumerate(X[self.transformation_target]):
                try:
                    words = word_tokenize(sentence)

                except TypeError:
                    words = []

                stemmed_words = []

                for word in words:
                    stemmed_words.append(ps.stem(word))

                stemmed_queries.append(" ".join(word for word in stemmed_words))

            X.loc[:, self.transformation_target] = stemmed_queries

        else:
            raise NotImplementedError("Unknown normalize_method")

        return X


class VectorizeQuery(BaseEstimator, TransformerMixin):
    """
    Prend le data frame qui contient au moins la colonne query_expression.
    Transforme chacun des mots en une colone de mot comptable avec selon une technique(vectorize_method)
    """

    def __init__(self, vectorize_method, freq_min=1, transformation_target="query_expression"):

        self.vectorize_method = vectorize_method
        self.freq_min = freq_min
        self.transformation_target = transformation_target

    def fit(self, X, y=None):
        self.update_class_vectorizer()
        queries = X[self.transformation_target].values.tolist()
        # Traitement particulier pour Word2Vec
        if self.vectorize_method == "Word2Vec":
            tokenized_queries = [word_tokenize(i) for i in queries]
            tokenized_queries = create_unk_tokens(tokenized_queries)
            if self.transformation_target == 'document_title':
                modelVec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
                self.vect = modelVec
            else:
                self.vect = Word2Vec(tokenized_queries, min_count=self.freq_min)

        else:
            self.vect.fit(queries)
        return self

    def update_class_vectorizer(self):

        if self.vectorize_method == "count":
            self.vect = CountVectorizer(min_df=self.freq_min)
        if self.vectorize_method == "binary count":
            self.vect = CountVectorizer(min_df=self.freq_min, binary=True)
        if self.vectorize_method == "tf-idf":
            self.vect = TfidfVectorizer(min_df=self.freq_min)
        if self.vectorize_method == "Word2Vec":
            self.vect = Word2Vec(min_count=self.freq_min)

    def transform(self, X):

        queries = X[self.transformation_target].values.tolist()
        if self.vectorize_method == "Word2Vec":
            vectorized_queries = []
            tokenized_queries = [word_tokenize(i) for i in queries]
            for sentence in tokenized_queries:
                word_vecs = []
                for word in sentence:
                    try:
                        word_vecs.append(self.vect[word])

                    except KeyError:
                        if self.transformation_target == "document_title":
                            word_vecs.append(self.vect["unknown"])
                        else:
                            word_vecs.append(self.vect["<UNK>"])

                vectorized_queries.append(np.mean(np.array(word_vecs), 0))

            df_vectorized_queries = pd.DataFrame(vectorized_queries)

        else:
            vectorized_queries = self.vect.transform(queries)

            df_vectorized_queries = pd.DataFrame(vectorized_queries.toarray(), columns=self.vect.get_feature_names())

        X = X.reset_index(drop=True)
        X = X.drop(columns=[self.transformation_target])
        if self.transformation_target == "query_expression":
            df_avec_nouvelles_valeurs = pd.concat([X, df_vectorized_queries], axis=1)
        else:
            df_avec_nouvelles_valeurs = pd.concat([X["document_id"], df_vectorized_queries], axis=1)

        return df_avec_nouvelles_valeurs


class TransformCategoricalVar(BaseEstimator, TransformerMixin):
    """
    Prends notre data frame X et converti nos variables catégoriques en numériques:
    user_country --> x0_Canada, x0_India,..... (0 ou 1)

    Transorme les NAN en "inconnu"
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.hot_encoder = OneHotEncoder(handle_unknown="ignore")
        X_cat = X.select_dtypes(include="object")
        X_cat = X_cat.fillna("inconnu")

        self.hot_encoder.fit(X_cat)
        # Pour noms des colonnes
        self.column_names = self.hot_encoder.get_feature_names()

        return self

    def transform(self, X):
        X = X.reset_index(drop=True)
        X_cat = X.select_dtypes(include="object")
        X_cat = X_cat.fillna("inconnu")
        array_cat = self.hot_encoder.transform(X_cat).toarray()

        data_frame_cat = pd.DataFrame(array_cat, columns=self.column_names)

        X_sans_cat = X.select_dtypes(exclude="object")

        df_avec_nouvelles_valeurs = pd.concat([X_sans_cat, data_frame_cat], axis=1)

        return df_avec_nouvelles_valeurs


########################################################################################################################
# Pas utile encore
########################################################################################################################
class TokenizeQuery(BaseEstimator, TransformerMixin):
    def __init__(self, tokenize_method):
        if tokenize_method not in ["word_tokenize"]:
            raise TypeError("{} is not a valid tokenizing method".format(tokenize_method))

        self.tokenize_method = tokenize_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.tokenize_method == "word_tokenize":
            X_trans = X.apply(lambda row: word_tokenize(row['query_expression']), axis=1)

        return X_trans

# A essayer ?
# http://www.davidsbatista.net/blog/2018/02/23/model_optimization/


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

        if self.current_name == "MLP":
            self.current_model_class = MLPClassifier()

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


def create_unk_tokens(tokenized_sentences):
    transformed_tokens = []
    list_of_known_words = []
    for sentence in tokenized_sentences:
        for i, word in enumerate(sentence):
            if word not in list_of_known_words:
                sentence[i] = "<UNK>"
                list_of_known_words.append(word)

        transformed_tokens.append(sentence)

    return transformed_tokens

