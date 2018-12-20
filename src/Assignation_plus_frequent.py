from import_data import import_raw_data,sophisticated_merge
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import  Counter
import numpy as np

def custom_scorer_mod(y,y_train):
    count_y=Counter(y_train)
    top5_list_count=count_y.most_common(5)
    top5_list=[]

    for el in top5_list_count:
        top5_list.append(el[0])

    top5_classes=np.asarray(top5_list)

    mask=np.isin(y,top5_classes)


    print("Score si assignation des plus fréquents",np.mean(mask))



def assignation(X,type):

    pass

# DATA
raw_data = import_raw_data()

df_searches_clicks_train = sophisticated_merge(raw_data["coveo_searches_train"],
                                               raw_data["coveo_clicks_train"],
                                               "merge_steph")

df_searches_clicks_valid = pd.merge(raw_data["coveo_searches_valid"],
                                    raw_data["coveo_clicks_valid"],
                                    on="search_id")

# LABELS
obj_labels_encoder = LabelEncoder()
labels_train = df_searches_clicks_train["document_id"].tolist()
labels_test = df_searches_clicks_valid["document_id"].tolist()
obj_labels_encoder.fit(labels_train+labels_test)

y_train = obj_labels_encoder.transform(labels_train)
y_valid = obj_labels_encoder.transform(labels_test)



#call functions
custom_scorer_mod(y_valid,y_train)