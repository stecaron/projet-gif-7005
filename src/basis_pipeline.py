import pandas as pd
from import_data import import_raw_data
from sklearn import pipeline
from pipeline_utils import FilterColumns, TokenizeQuery, VectorizeQuery, TransformCategoricalVar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from Score_thats_it import custom_scorer
import numpy as np


def main():
    raw_data = import_raw_data()

    df_searches_clicks = pd.merge(raw_data["coveo_searches_train"],
                                  raw_data["coveo_clicks_train"],
                                  on="search_id")

    model = pipeline.Pipeline([("data_extract", FilterColumns(filter_group="query_only")),
                               ("transform_data",
                                    pipeline.FeatureUnion([
                                        ("transformed_query",
                                            pipeline.Pipeline([("tokenize_query", TokenizeQuery(tokenize_method="word_tokenize")),
                                                               ("vectorize_query", VectorizeQuery(vectorize_method="count"))]))
                                    ])),
                               ("classifier", KNeighborsClassifier())])

    labels = df_searches_clicks["document_id"].tolist()
    obj_labels_encoder = LabelEncoder()
    y = obj_labels_encoder.fit_transform(labels)

    model.fit(df_searches_clicks, y)

def main2():

    #Data
    raw_data = import_raw_data()
    df_searches_clicks = pd.merge(raw_data["coveo_searches_train"],
                                  raw_data["coveo_clicks_train"],
                                  on="search_id")
    # Labels
    labels = df_searches_clicks["document_id"].tolist()
    obj_labels_encoder = LabelEncoder()
    y = obj_labels_encoder.fit_transform(labels)


    #POUR TEST BIDON
    mini_y= y[:1000]
    mini_df=df_searches_clicks[:1000]

    mini_y_test = y[1001:2000]
    mini_df_test = df_searches_clicks[1001:2000]



    #Pipeline de toutes les transformations qu'on fait, en ordre
    transformation_pipeline=pipeline.Pipeline([

        #("data_extract", FilterColumns(filter_group=["query_expression","search_nresults","user_language","user_country"])),
        ("data_extract", FilterColumns(filter_group=["query_expression","search_nresults"])),
        ("vectorize_query", VectorizeQuery(vectorize_method="count", freq_min=2)) #Tokenise deja je crois
        #("categorical_var_to_num",TransformCategoricalVar()) #Bug pour l'instant

     ])


    #Combine le transformer de data frame et le classifier
    final_pipe=pipeline.Pipeline([
        ("Transformer",transformation_pipeline),
        ("Classifier",LogisticRegression())
    ])


    #TEST BIDON
    X_essai_transformation=transformation_pipeline.fit_transform(mini_df)
    print(X_essai_transformation)

    X_test = transformation_pipeline.transform(mini_df_test)
    print(X_test)



    final_pipe.fit(mini_df,mini_y)
    print(final_pipe.score(mini_df,mini_y))
    print("Score Coveo:", custom_scorer(final_pipe, mini_df, mini_y))

    y_pred_deja_vu=final_pipe.predict(mini_df)

    y_pred=final_pipe.predict(mini_df_test)
    print(final_pipe.score(mini_df_test,mini_y_test))
    print("Score Coveo essaie de test:", custom_scorer(final_pipe, mini_df_test, mini_y_test))


    #print(final_pipe.get_params().keys())

    ####################################################################################################################
    #Optimisation avec Grid search
    ####################################################################################################################
    optimise=1
    if optimise==1:

        grille_finale={
            "Transformer__vectorize_query__freq_min": [1,2],
            "Transformer__vectorize_query__vectorize_method": ["count","tf-idf"],
            "Classifier__penalty":["l1","l2"]

        }
        grid_search=GridSearchCV(final_pipe,grille_finale,scoring=custom_scorer,cv=2)
        grid_search.fit(mini_df,mini_y)
        print("\n Grid search sur pipeline")
        print(grid_search.best_params_)
        print(grid_search.best_score_)





if __name__ == "__main__":
    #main()
    main2()