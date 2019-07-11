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


from pyids.ids_classifier import IDS, IDSOneVsAll
from pyids.ids_cacher import IDSCacher
from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_rule import IDSRule
from pyids.model_selection import encode_label, mode, CoordinateAscentOptimizer


df = pd.read_csv("./data/iris0.csv")
Y = df.iloc[:,-1]

"""
txns = TransactionDB.from_DataFrame(df)
rules = top_rules(txns.string_representation, appearance=txns.appeardict)
cars = createCARs(rules)
cars.sort(reverse=True)


ids_rules = map(IDSRule, cars[:20])
ids_ruleset = IDSRuleSet(ids_rules)
"""
quant_dataframe = QuantitativeDataFrame(df)



ids = IDSOneVsAll(quant_dataframe)
ids.fit()


print(ids.score_auc(quant_dataframe))

