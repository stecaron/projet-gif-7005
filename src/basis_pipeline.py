import pandas as pd
from import_data import import_raw_data,sophisticated_merge
from sklearn import pipeline
from pipeline_utils import FilterColumns, TokenizeQuery, VectorizeQuery, TransformCategoricalVar
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

from grid_search_utility import Make_All_Grid_Search_Models

#################################################
# Config
#################################################
config = {
        "Merge type": "basic",
        "Show test bidon": False,  # À enlever éventuellement

        "Create all grid searchs": {"Do it": False,
                                    "Save name": "basic"},

        "Show me all the grids": {"Do it": False,
                                  "Load name": "basic"},

        "Show me the best grid": {"Do it": True,
                                  "Load name": "basic"},
        "Export csv": True
        }

# Explications :
# "Merge type" ça permet de choisir le type de merge qu'on veut dans la fonction sophisticated_merge

# "Create all grid searchs" si "Do it" ==True, alors ça passe nos données dans tous les modèles et les test
# pour finalement sauvegarder les ensembles des de paramètres ainsi que leur score dans un fichier  Load name.p

# "Show me all the grids" si "Do it" ==True ça load le fichier et nous montre tous les essaies effectués

# "Show me the best grid" si "Do it" ==True ça load le modèle avec les meilleur paramètes à partir des meilleurs paramètres


# Utiliser skleanr v0.2
def main():

    raw_data = import_raw_data()

    df_searches_clicks_train = sophisticated_merge(raw_data["coveo_searches_train"],
                                                   raw_data["coveo_clicks_train"],
                                                   config["Merge type"])

    df_searches_clicks_valid = pd.merge(raw_data["coveo_searches_valid"],
                                        raw_data["coveo_clicks_valid"],
                                        on="search_id")

    # Labels
    obj_labels_encoder = LabelEncoder()
    labels_train = df_searches_clicks_train["document_id"].tolist()
    labels_test = df_searches_clicks_valid["document_id"].tolist()
    obj_labels_encoder.fit(labels_train+labels_test)

    y_train = obj_labels_encoder.transform(labels_train)
    y_valid = obj_labels_encoder.transform(labels_test)




    # Pour accélérer tests À RETIRER
    df_searches_clicks_train = df_searches_clicks_train[:2000]
    y_train = y_train[:2000]

    # Pipeline de toutes les transformations qu'on fait, en ordre
    transformation_pipeline = pipeline.Pipeline([

        ("data_extract", FilterColumns(filter_group=["query_expression",
                                                     "search_nresults",
                                                     "user_country",
                                                     "user_language"])),
        ("vectorize_query", VectorizeQuery(vectorize_method="count", freq_min=2)),
        ("categorical_var_to_num", TransformCategoricalVar())

     ])

    if config["Show test bidon"]:
        # TEST BIDON
        # POUR TEST, À RETIRER
        mini_y = y_train[:1000]
        mini_df = df_searches_clicks_train[:1000]

        mini_y_test = y_train[1001:2000]
        mini_df_test = df_searches_clicks_train[1001:2000]

        X_essai_transformation = transformation_pipeline.fit_transform(mini_df)
        print(X_essai_transformation)

        X_test = transformation_pipeline.transform(mini_df_test)
        print(X_test)

    ####################################################################################################################
    # Optimisation avec Grid search
    ####################################################################################################################

    grille_transformer = {
        "Transformer__vectorize_query__freq_min": [1, 2],
        "Transformer__vectorize_query__vectorize_method": ["count", "tf-idf"]
    }
    estimators = {
        "MLP": MLPClassifier(),
        # "XGB": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier()
    }
    grille_estimators = {
        "MLP": {"Classifier__activation": ["relu", "tanh"]},
        # "XGB": {"Classifier__n_estimators": [10, 32]},
        "KNN": {"Classifier__n_neighbors": [1, 3, 10, 15], "Classifier__weights": ["uniform", "distance"]}

    }

    Make_grid = Make_All_Grid_Search_Models(transformation_pipeline,
                                            grille_transformer,
                                            estimators,
                                            grille_estimators)

    if config["Create all grid searchs"]["Do it"]:
        Make_grid.test_best_grid_search(df_searches_clicks_train, y_train, config["Create all grid searchs"]["Save name"])

    if config["Show me all the grids"]["Do it"]:
        Make_grid.show_me_all_grids(config["Show me all the grids"]["Load name"])

    if config["Show me the best grid"]["Do it"]:

        final_pipe = Make_grid.return_best_pipeline(df_searches_clicks_train, y_train, config["Show me the best grid"]["Load name"])

        score_test = custom_scorer(final_pipe, df_searches_clicks_valid, y_valid)
        print("Score sur valid (utilisées comme test):", score_test)

        if config["Export csv"]:

            predict_top5_and_export_csv(final_pipe, raw_data["coveo_searches_test"], obj_labels_encoder)



    if config["Show test bidon"]:
        optimise_bidon = 0
        if optimise_bidon == 1:
            # Combine le transformer de data frame et le classifier
            final_pipe = pipeline.Pipeline([
                ("Transformer", transformation_pipeline),
                ("Classifier", MLPClassifier())
            ])

            grille_finale = {
                "Transformer__vectorize_query__freq_min": [1, 2],
                "Transformer__vectorize_query__vectorize_method": ["count", "tf-idf"],
                "Classifier__activation": ["relu", "tanh"]

            }
            grid_search = GridSearchCV(final_pipe, grille_finale, scoring=custom_scorer, cv=2)
            grid_search.fit(mini_df, mini_y)

            # Print
            print("\n Liste paramètres et scores")
            print("\n Grid search sur pipeline best:")
            print(grid_search.best_params_)
            print(grid_search.best_score_)


if __name__ == "__main__":
    main()