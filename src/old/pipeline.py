from sklearn import pipeline, model_selection
from pipeline_utils import CurrentModel, AddOtherFeatures, RemoveWords, NormalizeWords, VectorizeText
import pickle


def calculate_score():
    raise NotImplementedError()

def main():
    model = pipeline.Pipeline([("features",
                                pipeline.FeatureUnion(transformer_list=[("other_features", AddOtherFeatures(feature_to_add="None")),
                                                                        ("text_data", pipeline.Pipeline(
                                                                            ("remove_words", RemoveWords(words_to_remove="None"),
                                                                             "normalize_words", NormalizeWords(normalize_type="None"),
                                                                             "vectorize_text", VectorizeText(vectorize_type="None"),
                                                                             "reduce_dimension", ReduceDimension(reduction_type="None"),
                                                                             "normalize_features", NormalizeFeatures(normalize_type="None"))
                                                                        ))])),
                               ("classifier", CurrentModel(model_name="knn"))])

    scoring = calculate_score()

    grid_search_model = model_selection.GridSearchCV(
        model,
        {
            "features__other_features__feature_to_add": ["None"],
            "features__text_data__remove_words": ["None",
                                                  "tool_words",
                                                  "closed_class",
                                                  "tool_words_and_closed_class"],
            "features__text_data__normalize_words": ["None",
                                                     "Stemming",
                                                     "Lemmatization"],
            "features__text_data__vectorize_text": ["None",
                                                    "Presence",
                                                    "Frequency",
                                                    "td_idf"],
            "features__text_data__reduce_dimension": ["None",
                                                      "PCA"],
            "features__text_data__normalize_features": ["None",
                                                        "min_max_scale"]

        },
        n_jobs=-1,
        verbose=10,
        scoring=scoring,
        refit=False
    )

    grid_search_model.fit(X_train, y_train)
    pickle.dump(grid_search_model, open("fitted_pipeline", "wb"))


if __name__ == "__main__":
    main()