import logging

logging.basicConfig(level=logging.DEBUG)


from pyarc.data_structures import ClassAssocationRule
from pyids.data_structures import IDSRule
from pyids.model_selection.utils import encode_label
from sklearn.metrics import roc_auc_score
import numpy as np

import time
import pandas as pd
import pyarc
from pyids import IDS
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.rule_mining import RuleMiner
from pyarc.qcba.data_structures import QuantitativeDataFrame

from sklearn.model_selection import train_test_split


data = pd.read_csv("./data/titanic.csv").sample(frac=1).reset_index(drop=True)
data_train, data_test = train_test_split(data, test_size=0.8)

quant_dataframe_train = QuantitativeDataFrame(data_train)
quant_dataframe_test = QuantitativeDataFrame(data_test)

rm = RuleMiner()
rules = rm.mine_rules(data_train)

def train_ids(lambda_array):
    ids = IDS(
        algorithm="SLS",
    )
    ids.fit(
        class_association_rules=rules,
        quant_dataframe=quant_dataframe_train,
        lambda_array=lambda_array,
        optimizer_args=dict(
            omega_iterations=5000
        )
    )

    print("ids fitted")

    start_time = time.time()

    score_dict = dict()
    score_dict["acc_train"] = ids.score(quant_dataframe_train)
    score_dict["acc_test"] = ids.score(quant_dataframe_test)
    score_dict["auc_train_classbased"] = ids.score_auc(quant_dataframe_train, confidence_based=False)
    score_dict["auc_train_confbased"] = ids.score_auc(quant_dataframe_train, confidence_based=True)
    score_dict["auc_test_classbased"] = ids.score_auc(quant_dataframe_test, confidence_based=False)
    score_dict["auc_test_confbased"] = ids.score_auc(quant_dataframe_test, confidence_based=True)
    score_dict["duration"] = time.time()
    score_dict.update(ids.score_interpretability_metrics(quant_dataframe_test))

    return score_dict


lambda_arrays = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1000, 1000],
    [1, 1, 1, 1, 1, 1000, 1000],
    [500, 500, 1, 1, 1000, 1000, 1000],
]

benchmark_list = []

for lambda_array in lambda_arrays:
    bench_dict = train_ids(lambda_array)

    benchmark_list.append(bench_dict)

df = pd.DataFrame(benchmark_list)
df.to_csv("output_data/manual_lambda_tuning.csv")