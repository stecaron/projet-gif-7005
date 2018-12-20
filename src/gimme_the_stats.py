import pickle
import pandas as pd


load_name="Full_run"
list_all_grid_search = pickle.load(open(load_name+".p", "rb"))


#Data frame
df_mlp=pd.DataFrame(columns=["Score Valid","Layers shape","Normalisation","Type de vecteur","Frequence minimale"])
i_mlp=0

df_knn=pd.DataFrame(columns=["Score Valid","K","Type poids","Normalisation","Type de vecteur","Frequence minimale"])
i_knn=0


for dict_model in list_all_grid_search:
    name = dict_model["Model name"]
    score_moyen = dict_model["list Valid scores"]
    list_dict_params = dict_model["list Params dict"]



    for score, params in zip(score_moyen, list_dict_params):

        query_norm=params["Transformer__normalize_query__normalize_method"]
        vecto_method=params["Transformer__vectorize_query__vectorize_method"]
        vecto_freq_min=params["Transformer__vectorize_query__freq_min"]


        if name=="MLP":
            layers=params["Classifier__hidden_layer_sizes"]
            df_mlp.loc[i_mlp]=[score,layers,query_norm,vecto_method,vecto_freq_min]
            i_mlp+=1

        if name=="KNN":
            k=params["Classifier__n_neighbors"]
            weights=params["Classifier__weights"]
            df_knn.loc[i_knn]=[score,k,weights,query_norm,vecto_method,vecto_freq_min]
            i_knn+=1

# print("\nTableau MLP")
# print(df_mlp)
# print("\nTableau KNN")
# print(df_knn)

########################################################################################################################
#STATS TIME!!!!!!!!
########################################################################################################################
def mean_specific_var(model_name,df,var):
    print("\nModel {}".format(model_name))
    print(df.groupby(var).mean())

print(df_mlp.mean())
print(df_knn.mean())

mean_specific_var("MLP",df_mlp,"Normalisation")
mean_specific_var("MLP",df_mlp,"Type de vecteur")
mean_specific_var("MLP",df_mlp,"Frequence minimale")
mean_specific_var("MLP",df_mlp,"Layers shape")

mean_specific_var("KNN",df_knn,"Normalisation")
mean_specific_var("KNN",df_knn,"Type de vecteur")
mean_specific_var("KNN",df_knn,"Frequence minimale")
mean_specific_var("KNN",df_knn,"K")
mean_specific_var("KNN",df_knn,"Type poids")


#Meilleurs modeles
print("\n Meilleur modèle MLP")
print(df_mlp.loc[df_mlp["Score Valid"].idxmax()])
print("\n Meilleur modèle KNN")
print(df_knn.loc[df_knn["Score Valid"].idxmax()])
















