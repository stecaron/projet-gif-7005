import numpy as np

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


#TODO faire l'exportation csv
def predict_top5_and_export_csv(estimator,X):
    pass


