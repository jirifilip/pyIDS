import hyperopt
from hyperopt import fmin, tpe, hp, space_eval

import pandas as pd
import numpy as np
from pyids import IDS
from pyids.algorithms import mine_CARs
from pyids.algorithms.ids_multiclass import IDSOneVsAll
from pyids.data_structures import IDSRuleSet

from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import TransactionDB
from pyarc import CBA

import random
import logging
import time

import matplotlib.pyplot as plt

df = pd.read_csv("c:/code/python/machine_learning/assoc_rules/train/anneal0.csv")
quant_dataframe = QuantitativeDataFrame(df)


def is_solution_interpretable(metrics):
    return (
        metrics["fraction_overlap"] <= 0.10 and
        metrics["fraction_classes"] == 1.0 and
        metrics["fraction_uncovered"] <= 0.15 and
        metrics["average_rule_width"] < 8 and
        metrics["ruleset_length"] <= 10
    )


def objective(args):
    lambda_array = list(args.values())
    print(lambda_array)

    ids_multiclass = IDSOneVsAll(algorithm="RUSM")
    ids_multiclass.fit(quant_dataframe, lambda_array=lambda_array, rule_cutoff=200)

    metrics = ids_multiclass.score_interpretability_metrics(quant_dataframe)

    if not is_solution_interpretable(metrics):
        return 0

    auc = ids_multiclass.score_auc(quant_dataframe)
    print("AUC", auc)

    return -auc


space = {
    "lambda1": hp.uniform("l1", 0, 1000),
    "lambda2": hp.uniform("l2", 0, 10000000),
    "lambda3": hp.uniform("l3", 0, 500),
    "lambda4": hp.uniform("l4", 0, 500),
    "lambda5": hp.uniform("l5", 0, 10000000),
    "lambda6": hp.uniform("l6", 0, 10000000),
    "lambda7": hp.uniform("l7", 0, 10000000)
}

best = fmin(objective, space, algo=tpe.suggest, max_evals=500)

print(best)
