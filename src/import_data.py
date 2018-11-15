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

    return raw_data
