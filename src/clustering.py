import pandas as pd
from import_data import import_raw_data,sophisticated_merge
from sklearn import pipeline
from pipeline_utils import FilterColumns, TokenizeQuery, VectorizeQuery, TransformCategoricalVar, NormalizeQuery, RemoveStopWords
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from Score_thats_it import custom_scorer,predict_top5_and_export_csv
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.manifold import MDS, TSNE
from matplotlib import pyplot
import numpy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings

from grid_search_utility import Make_All_Grid_Search_Models
data=import_raw_data()
merge_type="merge_steph"

def merge_find_document_cluster(data , merge_type, n_clusters, algo_cluster):

    df_searches_clicks_train = sophisticated_merge(data["coveo_searches_train"],
                                                   data["coveo_clicks_train"],
                                                   merge_type)

    df_searches_clicks_train = df_searches_clicks_train.dropna(subset=['document_title'])

    df_searches_clicks_train_unique = df_searches_clicks_train.drop_duplicates(subset="document_id")
    y_stop_words = RemoveStopWords(df_searches_clicks_train_unique)

    normalize_query = NormalizeQuery(normalize_method="PorterStemmer", transformation_target="document_title")
    y_normalize = normalize_query.transform(y_stop_words)

    vectorize_query = VectorizeQuery(vectorize_method='tf-idf', transformation_target="document_title")
    vectorize_query.fit(y_normalize)
    y_vectorize = vectorize_query.transform(y_normalize)

    if algo_cluster == 'KMeans':
        y_model_cluster = KMeans(n_clusters=n_clusters, random_state=42)
        y_model_cluster.fit(y_vectorize.drop('document_id', axis=1))
        y_train_cluster = y_model_cluster.labels_.reshape(-1, 1)
    else:
        y_model_cluster = GaussianMixture(n_components=n_clusters, init_params='kmeans')
        y_train_cluster = y_model_cluster.fit_predict(y_vectorize.drop('document_id', axis=1)).reshape(-1, 1)
    document_id_array = numpy.array(df_searches_clicks_train_unique['document_id']).reshape(-1, 1)
    document_id_cluster = numpy.concatenate((document_id_array, y_train_cluster), axis=1)
    document_id_cluster_df = pd.DataFrame(data=document_id_cluster, columns=['document_id', 'document_cluster'])

    df_searches_clicks_train_cluster = pd.merge(df_searches_clicks_train, document_id_cluster_df, on='document_id', how = 'left')
    five_best_doc = {}
    for i in range(n_clusters):
        document_id_in_cluster = {}
        for index, row in df_searches_clicks_train_cluster.iterrows():
            if row['document_cluster'] == i:
                document_id_in_cluster[row['document_id']] = document_id_in_cluster.get(row['document_id'], 0) + 1
        five_best_doc[i] = [key for key in sorted(document_id_in_cluster, key=document_id_in_cluster.get, reverse=True)[:5]]
        if len(five_best_doc[i]) < 5:
            five_best_doc[i] = five_best_doc[i] + [''] * (5-len(five_best_doc[i]))
            five_best_doc[i] = np.array(five_best_doc[i], dtype=str)
    return df_searches_clicks_train_cluster, five_best_doc

#if __name__ == "__main__":
    #raw_data = import_raw_data()
    #test = merge_find_document_cluster(data=raw_data, merge_type="merge_steph", n_clusters=50, algo_cluster='KMeans')