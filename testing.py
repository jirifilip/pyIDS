import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.metrics import accuracy_score, auc, roc_auc_score

from pyids.ids_rule import IDSRule
from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_objective_function import ObjectiveFunctionParameters, IDSObjectiveFunction
from pyids.ids_optimizer import RSOptimizer, SLSOptimizer
from pyids.ids_cacher import IDSCacher
from pyids.ids_classifier import IDS


from pyarc.qcba import *


import cProfile




from pyarc.algorithms import createCARs, top_rules
from pyarc import TransactionDB


df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/titanic.csv")
Y = df.iloc[:,-1]
txns = TransactionDB.from_DataFrame(df)
rules = top_rules(txns.string_representation, appearance=txns.appeardict)
cars = createCARs(rules)


quant_df = QuantitativeDataFrame(df)

ids = IDS()
ids.fit(quant_df, cars[:40], algorithm="SLS")

acc = ids.score_auc(quant_df)

print(acc)