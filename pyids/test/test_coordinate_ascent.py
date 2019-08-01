import unittest
import pandas as pd
import random

from pyids.ids_classifier import IDS, mine_IDS_ruleset
from pyids.model_selection import CoordinateAscentOptimizer
from pyarc.qcba.data_structures import QuantitativeDataFrame

class TestCoordinateAscent(unittest.TestCase):

    def test_optimization(self):
        df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/titanic.csv")
        ids_ruleset = mine_IDS_ruleset(df, rule_cutoff=40)

        quant_df = QuantitativeDataFrame(df)
        ascent = CoordinateAscentOptimizer(IDS(), maximum_consecutive_iterations=1)
        lambdas = ascent.fit(ids_ruleset, quant_df, quant_df)

