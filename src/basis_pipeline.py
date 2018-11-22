import pandas as pd
from import_data import import_raw_data
from sklearn import pipeline
from pipeline_utils import FilterColumns, TokenizeQuery, VectorizeQuery
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    raw_data = import_raw_data()

    df_searches_clicks = pd.merge(raw_data["coveo_searches_train"],
                                  raw_data["coveo_clicks_train"],
                                  on="search_id")

    model = pipeline.Pipeline([("data_extract", FilterColumns(filter_group="query_only")),
                               ("transform_data",
                                    pipeline.FeatureUnion([
                                        ("transformed_query",
                                            pipeline.Pipeline([("tokenize_query", TokenizeQuery()),
                                                               ("vectorize_query", VectorizeQuery(vectorize_method="count"))]))
                                    ])),
                               ("classifier", KNeighborsClassifier())])

    labels = df_searches_clicks["document_id"].tolist()
    obj_labels_encoder = LabelEncoder()
    y = obj_labels_encoder.fit_transform(labels)

    model.fit(df_searches_clicks[["query_expression"]], y)



if __name__ == "__main__":
    main()