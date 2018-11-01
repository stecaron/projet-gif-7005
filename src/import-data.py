#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 06:46:01 2018

@author: stephanecaron
"""

import pandas

coveo_clicks_train = pandas.read_csv('data/coveo_clicks_train.csv')
coveo_clicks_valid = pandas.read_csv('data/coveo_clicks_valid.csv')
coveo_searches_train = pandas.read_csv('data/coveo_searches_train.csv')
coveo_searches_valid = pandas.read_csv('data/coveo_searches_valid.csv')