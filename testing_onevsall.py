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
df_test = pd.read_csv("./data/iris1.csv")
Y = df.iloc[:,-1]

quant_dataframe = QuantitativeDataFrame(df)
quant_dataframe_test = QuantitativeDataFrame(df_test)


ids = IDSOneVsAll()
ids.fit(quant_dataframe)


print(ids.score_auc(quant_dataframe_test))

