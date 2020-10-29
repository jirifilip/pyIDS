import os
import re
import pandas as pd
import logging
import numpy as np

from pyarc.qcba.data_structures import QuantitativeDataFrame
import time

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.data_structures.ids_ruleset import IDSRuleSet
from pyids.algorithms.ids_classifier import IDSClassifier
from pyids.algorithms.ids import IDS
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import M1Algorithm
from pyarc import CBA

#logging.basicConfig(level=logging.DEBUG)

datasets = [
    "iris",
    "breast-w",
    "anneal",
    "hypothyroid",
    "ionosphere",
    "lymph",
    "vehicle",
    "autos",
    "diabetes",
    "glass",
    "heart-h",
    "tic-tac-toe",
    "australian",
    "sick",
    "segment",
    "spambase",
    "sonar",
    "vowel",
    "hepatitis",
    "credit-a",
    "mushroom",
    "house-votes-84",
    "soybean",
    "primary-tumor",
    "credit-g",
    "audiology",
    "breast-cancer",
    "balance-scale",
    "heart-c",
    "kr-vs-kp",
    "pima",
    "heart-statlog"]

dataset_files = [ f"{dataset_name}0.csv" for dataset_name in datasets ]

dataset_path = "C:/code/python/machine_learning/assoc_rules/"
dataset_path_train = os.path.join(dataset_path, "train")
dataset_path_test = os.path.join(dataset_path, "test")

benchmark_list = []

for dataset_filename in dataset_files:
    print(dataset_filename)

    df_train = pd.read_csv(os.path.join(dataset_path_train, dataset_filename))
    df_test = pd.read_csv(os.path.join(dataset_path_test, dataset_filename))

    txns_train = TransactionDB.from_DataFrame(df_train)
    txns_test = TransactionDB.from_DataFrame(df_test)

    quant_df_train = QuantitativeDataFrame(df_train)
    quant_df_test = QuantitativeDataFrame(df_test)

    cba = CBA(support=0.1, confidence=0.1)
    cba.fit(txns_train)

    rules = cba.clf.rules
    ids_ruleset = IDSRuleSet.from_cba_rules(rules)

    ids = IDS()
    ids.clf = IDSClassifier(ids_ruleset.ruleset)
    ids.clf.default_class = cba.clf.default_class

    metrics_dict = ids.score_interpretability_metrics(quant_df_test)

    benchmark_dict = dict(
        dataset_filename=dataset_filename,
        algorithm="cba"
    )

    benchmark_dict.update(metrics_dict)
    print(benchmark_dict)

    benchmark_list.append(benchmark_dict)


benchmark_df = pd.DataFrame(benchmark_list)
benchmark_df.to_csv("output_data/cba_interpretability_benchmark.csv", index=False)