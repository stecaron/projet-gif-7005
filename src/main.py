#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:35:36 2018

@author: Samuel_Levesque
"""

from run_config import run_config
from import_data import import_raw_data


def main():
    if run_config["import_data"]:
        raw_data = import_raw_data()

    if run_config["match_features"]:
        pass

    if run_config["vectorize_features"]:
        pass

    if run_config["extract_documents_features"]:
        pass

    if run_config["fit_knn"]:
        pass

    if run_config["fit_naive_bayes"]:
        pass

    if run_config["fit_svm"]:
        pass

    if run_config["fit_nn"]:
        pass


if __name__ == "__main__":
    main()
