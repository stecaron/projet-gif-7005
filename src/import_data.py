#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 06:46:01 2018

@author: stephanecaron
"""

import pandas as pd

def import_raw_data():
    raw_data = {}

    raw_data["coveo_clicks_train"] = pd.read_csv(r"../data/coveo_clicks_train.csv")
    raw_data["coveo_clicks_valid"] = pd.read_csv(r"../data/coveo_clicks_valid.csv")
    raw_data["coveo_searches_train"] = pd.read_csv(r"../data/coveo_searches_train.csv")
    raw_data["coveo_searches_valid"] = pd.read_csv(r"../data/coveo_searches_valid.csv")
    raw_data["coveo_searches_test"]=pd.read_csv(r"../data/coveo_searches_test.csv")
    
    return raw_data



def sophisticated_merge(df_searchs,df_clicks,merge_name):
    '''

    Merge les clicks au searchs en se basant sur le search_id.
    Trouver une façon intelligente d'assigner des documents_id au search qui n'ont résulté en un click

    :param df_searchs:
    :param df_clicks:
    :param merge_name: (str) nom par lequel on désigne le type de merge
    :return: Un data frame pandas avec toutes les colones contenus dans df_searchs et df_clicks
    '''

    if merge_name=="basic":
        df_searches_clicks_train = pd.merge(df_searchs,df_clicks,on="search_id")


    elif merge_name=="merge_steph":

        # On garde seulement les clicks finals pour une search (basé sur click_datetime)
        idx = df_clicks.groupby(['search_id'])['click_datetime'].transform(max) == df_clicks['click_datetime']
        df_clicks = df_clicks[idx]

        # On merge les searchs sur les clicks
        df_searches_clicks_train = pd.merge(df_searchs,
                                            df_clicks,
                                            on="search_id",
                                            how='left')

        # On enleve les searchs sans clicks
        df_searches_clicks_train = df_searches_clicks_train.dropna(subset=['click_datetime'])
    


    return df_searches_clicks_train


#Colonne dans search:
#search_id	search_datetime	search_cause	search_nresults	query_expression	query_pipeline	facet_title	facet_value	visit_id	visitor_id	user_id	user_language	user_device	user_is_mobile	user_country	user_city	user_region	user_type
