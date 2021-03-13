import unittest
import os
import pandas as pd

from utils import resource_path

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.algorithms.classifier import mine_CARs
from pyids.algorithms.objective_function import NormalizedF1ObjectiveFunction
from pyids import IDS


class TestIDS(unittest.TestCase):

    def test_ids_fit(self):
        iris_path = os.path.join(resource_path, "iris0.csv")
        df = pd.read_csv(iris_path)
        quant_df = QuantitativeDataFrame(df)

        cars = mine_CARs(df, rule_cutoff=20)

        ids = IDS()
        ids.fit(dataframe=quant_df, rules=cars)

    def test_ids_with_different_objective(self):
        iris_path = os.path.join(resource_path, "iris0.csv")
        df = pd.read_csv(iris_path)
        quant_df = QuantitativeDataFrame(df)

        cars = mine_CARs(df, rule_cutoff=20)

        objective = NormalizedF1ObjectiveFunction(
            dataframe=quant_df,
            rules=cars
        )

        ids = IDS()
        ids.fit(dataframe=quant_df, rules=cars, objective=objective)

    def test_ids_auc_calculation(self):
        iris_path = os.path.join(resource_path, "iris0.csv")
        df = pd.read_csv(iris_path)
        quant_df = QuantitativeDataFrame(df)

        cars = mine_CARs(df, rule_cutoff=20)

        ids = IDS()
        ids.fit(dataframe=quant_df, rules=cars)
        ids.score_auc(quant_df)