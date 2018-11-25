from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from word_to_remove_factory import WordsToRemoveFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

#Ancienne version pour le main original
# class FilterColumns(BaseEstimator, TransformerMixin):
#     def __init__(self, filter_group):
#
#         if filter_group not in ["query_only"]:
#             raise TypeError("{} is not a valid filter_group".format(filter_group))
#
#         self.filter_group = filter_group
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         if self.filter_group == "query_only":
#             X = X[["query_expression"]]
#
#         return X


class FilterColumns(BaseEstimator, TransformerMixin):
    '''
    filter_group : list des noms de columns du data frame qu'on veut conserver

    '''
    def __init__(self, filter_group):
        self.filter_group = filter_group

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X[self.filter_group]

        return X



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


# class VectorizeQuery(BaseEstimator, TransformerMixin):
#     def __init__(self, vectorize_method):
#         if vectorize_method not in ["count"]:
#             raise TypeError("{} is not a valid vectorize method".format(vectorize_method))
#
#         self.vectorize_method = vectorize_method
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         if self.vectorize_method == "count":
#             vect = CountVectorizer()
#             queries = X.values.tolist()
#
#             vectorized_queries = vect.fit_transform(queries)
#
#         return vectorized_queries.toarray()


# class VectorizeQuery(BaseEstimator, TransformerMixin):
#     '''
#     Prend le data frame qui contient au moins la colonne query_expression.
#     Transforme chacun des mots en une colone de mot comptable avec selon une technique(vectorize_method)
#     '''
#     def __init__(self, vectorize_method,freq_min=1):
#
#         self.vectorize_method = vectorize_method
#         self.freq_min =freq_min
#
#     def fit(self, X, y=None):
#
#         return self
#
#     def transform(self, X):
#         if self.vectorize_method == "count":
#             vect = CountVectorizer(min_df=self.freq_min)
#         if self.vectorize_method == "binary count":
#             vect = CountVectorizer(min_df=self.freq_min,binary=True)
#
#
#
#
#         queries = X["query_expression"].values.tolist()
#         vectorized_queries = vect.fit_transform(queries)
#
#         df_vectorized_queries=pd.DataFrame(vectorized_queries.toarray(),columns=vect.get_feature_names())
#
#         X=X.drop(columns=["query_expression"])
#
#         return pd.concat([X,df_vectorized_queries],axis=1)

class VectorizeQuery(BaseEstimator, TransformerMixin):
    '''
    Prend le data frame qui contient au moins la colonne query_expression.
    Transforme chacun des mots en une colone de mot comptable avec selon une technique(vectorize_method)
    '''

    def __init__(self, vectorize_method, freq_min=1):

        self.vectorize_method = vectorize_method
        self.freq_min = freq_min

    def fit(self, X, y=None):
        self.update_class_vectorizer()
        queries=X["query_expression"].values.tolist()
        self.vect.fit(queries)
        return self



    def update_class_vectorizer(self):

        if self.vectorize_method == "count":
            self.vect = CountVectorizer(min_df=self.freq_min)
        if self.vectorize_method == "binary count":
            self.vect = CountVectorizer(min_df=self.freq_min, binary=True)
        if self.vectorize_method == "tf-idf":
            self.vect = TfidfVectorizer(min_df=self.freq_min)


    def transform(self, X):


        queries = X["query_expression"].values.tolist()
        vectorized_queries = self.vect.transform(queries)

        df_vectorized_queries = pd.DataFrame(vectorized_queries.toarray(), columns=self.vect.get_feature_names())

        X=X.reset_index(drop=True)
        X = X.drop(columns=["query_expression"])

        df_avec_nouvelles_valeurs=pd.concat([X, df_vectorized_queries], axis=1)

        return df_avec_nouvelles_valeurs



class TransformCategoricalVar(BaseEstimator,TransformerMixin):
    '''
    Prends notre data frame X et converti nos variables catégoriques en numériques:
    user_country --> user_country_Canada, user_country_India,..... (0 ou 1)

    BUG A RÉGLER: faire en sorte que le fit conserve seulement les variables vue en fit
    '''
    def __init__(self):
        pass


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X=pd.get_dummies(X)

        return X



########################################################################################################################


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


#
# class NormaliseWords(BaseEstimator, TransformerMixin):
#     def __init__(self, normalise_type):
#         self.normalise_type = normalise_type
#
#         if self.normalise_type == "lemmatize" or "stemming" or "none":
#             pass
#         else:
#             raise TypeError("{} is not lemmatize, stemming or none".format(self.normalise_type))
#
#     def fit(self, x, y=None):
#         return self
#
#     def transform(self, x):
#         if self.normalise_type == "none":
#             x = np.array(list(x))
#         elif self.normalise_type == "lemmatize":
#             x = map(lambda r: ' '.join([wordnet.WordNetLemmatizer().lemmatize(i.lower()) for i in r.split()]), x)
#             x = np.array(list(x))
#         elif self.normalise_type == "stemming":
#             x = map(lambda r: ' '.join([porter.PorterStemmer().stem(i.lower()) for i in r.split()]), x)
#             x = np.array(list(x))
#         return x