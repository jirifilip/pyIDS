import unittest
import pandas as pd
import random

from pyids.ids_classifier import IDS, mine_CARs
from pyarc.qcba.data_structures import QuantitativeDataFrame

class TestIDSClassifier(unittest.TestCase):

    def test_model_fitting(self):
        df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/titanic.csv")
        cars = mine_CARs(df, rule_cutoff=40)

        quant_df = QuantitativeDataFrame(df)
        ids = IDS()
        ids.fit(quant_df, cars, debug=False)
        auc = ids.score_auc(quant_df)

    def test_sls_algorithm(self):
        df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/titanic.csv")
        cars = mine_CARs(df, rule_cutoff=40)

        quant_df = QuantitativeDataFrame(df)
        ids = IDS()
        ids.fit(quant_df, cars, algorithm="SLS", debug=False)
        auc = ids.score_auc(quant_df)

    def test_dls_algorithm(self):
        df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/titanic.csv")
        cars = mine_CARs(df, rule_cutoff=40)

        quant_df = QuantitativeDataFrame(df)
        ids = IDS()
        ids.fit(quant_df, cars, algorithm="DLS", debug=False)
        auc = ids.score_auc(quant_df)
