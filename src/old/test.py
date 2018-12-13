import pandas as pd
from import_data import import_raw_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


dict_raw_data=import_raw_data()
df_searches=dict_raw_data["coveo_searches_train"]
df_clicks=dict_raw_data["coveo_clicks_train"]

df_searches_clicks=pd.merge(df_clicks,df_searches,on="search_id")
print(df_searches_clicks)
#print(df_clicks)
#print(df_searches)



#Labels
labels=df_searches_clicks["document_id"].tolist()
obj_labels_encoder =LabelEncoder()
y=obj_labels_encoder.fit_transform(labels)

#Corpus querry
corpus=df_searches_clicks[["query_expression"]]

corpus=corpus["query_expression"].tolist()

#Split
corpus_train,corpus_test,y_train,y_test =train_test_split(corpus,y,train_size=0.7,random_state=123)

#Objet de compte
objCount=CountVectorizer(min_df=20)
objCount.fit(corpus_train)
print(objCount.get_feature_names())

#Mettre .toarray() ???
X_train=objCount.transform(corpus_train).toarray()
X_test=objCount.transform(corpus_test).toarray()



test=1
if test==1:
    clf=KNeighborsClassifier(5)
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)
    print("Score kppv",score)
    #k_neighbor=clf.kneighbors(X_test,n_neighbors=5)
    #print(k_neighbor)

    clf_reg_log=LogisticRegression()
    clf_reg_log.fit(X_train,y_train)
    score=clf_reg_log.score(X_test,y_test)
    print("Score log reg",score)
    print(clf_reg_log.predict_proba(X_test))
    print(clf_reg_log.predict_proba(X_test[0,:]))
