import numpy as np
import pandas as pd
from sklearn import pipeline


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


#TODO trouver la source du problème. TROUVÉ !
def predict_top5_and_export_csv(estimator,X,obj_label):

    #TESTING À RETIRER
    pipeline_transformation_seulement=pipeline.Pipeline(estimator.steps[:-1])#Pogne la pipeline, sans la classification
    print(pipeline_transformation_seulement.transform(X)) #La transformation du data set complet se fait bien
    #X=X[:440] #plus haute valeur bug, car manque données dans search_nresults
    #FIN TESTING

    proba_ordered_by_classes = estimator.predict_proba(X)

    ordered_classes = estimator.classes_
    best_proba_order = np.argsort(proba_ordered_by_classes)

    best_classes = ordered_classes[best_proba_order]
    top5_classes = best_classes[:, -5:]


    #Convertion labels et exportation
    data={}
    data["search_id"]=X["search_id"]
    j=1
    for i in range(4,-1,-1):
        test=top5_classes[:,i]
        data["doc{}".format(j)]=obj_label.inverse_transform(test)
        j+=1


    frame=pd.DataFrame(data)
    frame.to_csv("predictions.csv", sep=',', encoding='utf-8',index=False)
    print("Fichier exporté sous le nom: predictions.csv")



