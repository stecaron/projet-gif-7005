from sklearn.model_selection import GridSearchCV
from Score_thats_it import custom_scorer
from sklearn import pipeline
import pickle

# https://stackoverflow.com/questions/31948879/using-explict-predefined-validation-set-for-grid-search-with-sklearn


class Make_All_Grid_Search_Models():
    def __init__(self, transformation_pipeline, transformation_grid, estimators, estimator_grid):
        """
        :param transformation_pipeline: Notre objet pipeline de transformation
        :param transformation_grid: dictionnaire de de paramètre pour la grille de transformation
        :param estimators: dictionnaire avec nos modèle de classification
        :param estimator_grid: ictionnaire de de paramètre pour la grille des modèles de classification
        """
        self.transformation_pipeline = transformation_pipeline
        self.transformation_grid = transformation_grid
        self.estimators = estimators
        self.estimator_grid = estimator_grid

    def test_best_grid_search(self, X, y, save_name):
        """
        Crée le fichier list_grid_search.p qui contient un dictionnaire de paramètres pour chaque classificateur
        :param y:
        :return:
        """
        list_all_grid_search = []
        for key in self.estimators.keys():
            final_grid = {**self.transformation_grid, **self.estimator_grid[key]}

            final_pipe = pipeline.Pipeline([
                ("Transformer", self.transformation_pipeline),
                ("Classifier", self.estimators[key])
            ])
            print("Testing for classifier {}".format(key))
            grid_search = GridSearchCV(final_pipe, final_grid,  cv=2, scoring=custom_scorer)
            grid_search.fit(X, y)

            scores_moyens = grid_search.cv_results_['mean_test_score']
            list_dict_params = grid_search.cv_results_['params']
            classifier_name = key

            dict_model = {"Model name": classifier_name, "list Valid scores": scores_moyens,
                          "list Params dict": list_dict_params}

            list_all_grid_search.append(dict_model)

        pickle.dump(list_all_grid_search, open(save_name+".p", "wb"))

    def return_best_pipeline(self, X, y, load_name):
        """
        À partir de la liste du fichier list_grid_search.p, on prend les paramètres et on crée la pipeline avec les
        paramètres ayant le mieux performés
        :return:
        """
        list_all_grid_search = pickle.load(open(load_name+".p", "rb"))

        best_score = 0
        for dict_model in list_all_grid_search:
            name = dict_model["Model name"]
            score_moyen = dict_model["list Valid scores"]
            list_dict_params = dict_model["list Params dict"]

            for score, params in zip(score_moyen, list_dict_params):

                if score > best_score:
                    best_score = score
                    best_dict_model = {"Model name": name, "Score valid": score, "Params dict": params}

        print("\nBest model:")
        print(best_dict_model)

        final_pipe = pipeline.Pipeline([
            ("Transformer", self.transformation_pipeline),
            ("Classifier", self.estimators[best_dict_model["Model name"]])
        ])

        final_pipe.set_params(**best_dict_model["Params dict"])
        final_pipe.fit(X, y)

        return final_pipe

    def show_me_all_grids(self, load_name):
        """
        affiche toutes les combinaisons de paramètres et leur scores
        :return:
        """
        list_all_grid_search = pickle.load(open(load_name+".p", "rb"))

        for dict_model in list_all_grid_search:
            name = dict_model["Model name"]
            score_moyen = dict_model["list Valid scores"]
            list_dict_params = dict_model["list Params dict"]

            for score, params in zip(score_moyen, list_dict_params):
                print("Model {}, Score valid :{}, params:{}".format(name, score, params))
