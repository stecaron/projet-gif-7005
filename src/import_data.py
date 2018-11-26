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


#TODO à faire
def sophisticated_merge(df_searchs,df_clicks):
    '''
    Merge les clicks au searchs en se basant sur le search_id.
    Trouver une façon intelligente d'assigner des documents_id au search qui n'ont résulté en un click

    :param df_searchs:
    :param df_clicks:
    :return: Un data frame pandas avec toutes les colones contenus dans df_searchs et df_clicks
    '''
    pass


#Colonne dans search:
#search_id	search_datetime	search_cause	search_nresults	query_expression	query_pipeline	facet_title	facet_value	visit_id	visitor_id	user_id	user_language	user_device	user_is_mobile	user_country	user_city	user_region	user_type
