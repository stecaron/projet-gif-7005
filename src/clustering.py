import pandas as pd
from import_data import import_raw_data,sophisticated_merge
from sklearn import pipeline
from pipeline_utils import FilterColumns, TokenizeQuery, VectorizeQuery, TransformCategoricalVar, NormalizeQuery
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


from grid_search_utility import Make_All_Grid_Search_Models

raw_data = import_raw_data()

df_searches_clicks_train = sophisticated_merge(raw_data["coveo_searches_train"],
                                               raw_data["coveo_clicks_train"],
                                               "merge_steph")

normalize_query = NormalizeQuery(normalize_method="PorterStemmer", transformation_target="document_title")
y_normalize = normalize_query.transform(df_searches_clicks_train)

