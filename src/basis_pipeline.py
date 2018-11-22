import pandas as pd
from import_data import import_raw_data
from sklearn import pipeline
from pipeline_utils import CurrentModel, AddOtherFeatures,  RemoveWords

def main():
    raw_data = import_raw_data()

    df_searches_clicks = pd.merge(raw_data["coveo_searches_train"],
                                  raw_data["coveo_clicks_train"],
                                  on="search_id")

    model = pipeline.Pipeline([("features",
                                pipeline.FeatureUnion(transformer_list=[("other_features", AddOtherFeatures(feature_to_add="None")),
                                                                        ("text_data", pipeline.Pipeline(
                                                                            ("remove_words", RemoveWords(words_to_remove="None"))
                                                                        ))])),
                               ("classifier", CurrentModel(model_name="knn"))])

    print(df_searches_clicks)



if __name__ == "__main__":
    main()