import pandas as pd
import numpy as np


#Ne pas utiliser
def score_from_data_frame(df_pred,df_real):

    """
    Calcul le score d'évaluation que covéo utilise
    :param df_pred: data_frame pandas sous la forme search_id, doc1, doc2, doc3, doc3, doc4, doc5
    :param df_real: data_frame pandas sous la form search_id doc
    :return: score float
    """

    df_searches_clicks = pd.merge(df_pred, df_real, on="search_id")



    df_searches_clicks["is in 1"] =np.where(df_searches_clicks["doc"].isin(df_searches_clicks["doc1"]),1,0)
    df_searches_clicks["is in 2"] = np.where(df_searches_clicks["doc"].isin(df_searches_clicks["doc2"]), 1, 0)
    df_searches_clicks["is in 3"] = np.where(df_searches_clicks["doc"].isin(df_searches_clicks["doc3"]), 1, 0)
    df_searches_clicks["is in 4"] = np.where(df_searches_clicks["doc"].isin(df_searches_clicks["doc4"]), 1, 0)
    df_searches_clicks["is in 5"] = np.where(df_searches_clicks["doc"].isin(df_searches_clicks["doc5"]), 1, 0)
    df_searches_clicks["Reussi"]= df_searches_clicks[["is in 1","is in 2","is in 3","is in 4","is in 5",]].max(axis=1)


    moyenne=df_searches_clicks["Reussi"].mean()



    print(df_searches_clicks)
    print(moyenne)
    return moyenne

def custom_scorer(estimator,X,y):
    """
    À partir d'un classificateur qui a la méthode predict_proba, de X et de y, on calcul le score de coveo
    :param estimator:
    :param X:
    :param y:
    :return:
    """

    #On pourrait modifier ça si on utilise un clf qui n'a pas la fonction
    proba_ordered_by_classes=estimator.predict_proba(X)


    ordered_classes=estimator.classes_
    best_proba_order=np.argsort(proba_ordered_by_classes)

    best_classes=ordered_classes[best_proba_order]
    top5_classes=best_classes[:,-5:]

    def fn(x):
        return x==y


    mat_bool=np.apply_along_axis(fn,axis=0,arr=top5_classes)

    return np.mean(np.apply_along_axis(max,axis=1,arr=mat_bool))



if __name__ == "__main__":

    df1 = pd.DataFrame({'doc': ['a', 'b', 't', 'd'],"search_id":[2,1,3,4]})
    df2 = pd.DataFrame({'doc1': ['g','w','r', 't'],'doc2': ['a','a','s', 't'],'doc3': ['a','z','t', 'g'],'doc4': ['b','f','t', 't'],'doc5': ['a','g','t', 'm'],"search_id":[1,2,4,3]})
    score_from_data_frame(df2,df1)



