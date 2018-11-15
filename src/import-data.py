#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 06:46:01 2018

@author: stephanecaron
"""

import pandas
import numpy as np

X_full_data = np.genfromtxt("../data/coveo_clicks_train.csv", delimiter=",", skip_header=True)

coveo_clicks_train = pandas.read_csv(r"../data/coveo_clicks_train.csv")
coveo_clicks_valid = pandas.read_csv('data/coveo_clicks_valid.csv')
coveo_searches_train = pandas.read_csv('data/coveo_searches_train.csv')
coveo_searches_valid = pandas.read_csv('data/coveo_searches_valid.csv')