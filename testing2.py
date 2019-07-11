from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import scipy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import Counter


from pyarc.qcba import *

from pyarc.algorithms import createCARs, top_rules
from pyarc import TransactionDB

from pyids import IDS
from pyids.ids_cacher import IDSCacher
from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_rule import IDSRule
from pyids.model_selection import encode_label, mode


df = pd.read_csv("./data/titanic.csv")

txns = TransactionDB.from_DataFrame(df)
rules = top_rules(txns.string_representation, appearance=txns.appeardict)
cars = createCARs(rules)

quant_dataframe = QuantitativeDataFrame(df)

cutoff_subset = 60
cars_subset = cars[:cutoff_subset]
ids_rls_subset = map(IDSRule, cars_subset)
ids_ruleset = IDSRuleSet(ids_rls_subset)

ids_cacher = IDSCacher()
ids_cacher.calculate_overlap(ids_ruleset, quant_dataframe)


l_arr = [762, 902, 837, 950, 958, 563, 962]

iters = 10

ids = IDS()
ids.cacher = ids_cacher
ids.ids_ruleset = ids_ruleset
auc_scores = []
for i in range(iters):
    ids.fit(quant_dataframe, debug=False, lambda_array=l_arr)
    score = ids.score_auc(quant_dataframe)
    print("score:", score)
    auc_scores.append(score)

print("max score", max(auc_scores))


iters = 10

auc_scores = []
for i in range(iters):
    ids.fit(quant_dataframe, debug=False)
    score = ids.score_auc(quant_dataframe)
    auc_scores.append(score)

print(max(auc_scores))