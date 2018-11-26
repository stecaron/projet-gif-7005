from sklearn.model_selection import GridSearchCV
from Score_thats_it import custom_scorer
from sklearn import pipeline
import pickle


# https://stackoverflow.com/questions/31948879/using-explict-predefined-validation-set-for-grid-search-with-sklearn


#Je sais pas pourquoi j'ai fait une classe avec ça, c'était pas nécessaire
class Make_All_Grid_Search_Models():
    def __init__(self,transformation_pipeline,transformation_grid,estimators,estimator_grid):
        '''
        :param transformation_pipeline: Notre objet pipeline de transformation
        :param transformation_grid: dictionnaire de de paramètre pour la grille de transformation
        :param estimators: dictionnaire avec nos modèle de classification
        :param estimator_grid: ictionnaire de de paramètre pour la grille des modèles de classification
        '''
        self.transformation_pipeline= transformation_pipeline
        self.transformation_grid =transformation_grid
        self.estimators =estimators
        self.estimator_grid =estimator_grid

    def test_best_grid_search(self,X,y):
        '''
        Crée le fichier list_grid_search.p qui contient une liste de grid search pour chaque classificateur
        :param X: data frame raw
        :param y:
        :return:
        '''
        list_all_grid_search=[]
        for key in self.estimators.keys():
            final_grid={**self.transformation_grid,**self.estimator_grid[key]}

            final_pipe = pipeline.Pipeline([
                ("Transformer", self.transformation_pipeline),
                ("Classifier", self.estimators[key])
            ])
            print("Testing for classifier {}".format(key))
            grid_search=GridSearchCV(final_pipe,final_grid,cv=2,scoring=custom_scorer)
            grid_search.fit(X,y)

            list_all_grid_search.append(grid_search)

        pickle.dump(list_all_grid_search, open("list_grid_search.p", "wb"))



    def return_best_grid_search(self):
        '''
        À partir de la liste du fichier list_grid_search.p renvoit le modèle grid search avec le meilleur score
        :return:
        '''
        list_all_grid_search=pickle.load( open( "list_grid_search.p", "rb" ) )
        best_score=0
        for grid_search in list_all_grid_search:
            score=grid_search.best_score_

            if score>best_score:
                best_grid_search=grid_search

        return best_grid_search



    def show_me_all_grids(self):
        """
        affiche toutes les combinaisons de paramètres et leur scores
        :return:
        """
        list_all_grid_search = pickle.load(open("list_grid_search.p", "rb"))

        for grid_search in list_all_grid_search:
            scores_moyens=grid_search.cv_results_['mean_test_score']
            dict_params=grid_search.cv_results_['params']
            for score,params in zip(scores_moyens,dict_params):
                print("Score :{} avec {}".format(score,params))





