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


from pyids.ids_classifier import IDS, mine_IDS_ruleset
from pyids.ids_cacher import IDSCacher
from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_rule import IDSRule
from pyids.model_selection import encode_label, mode, CoordinateAscentOptimizer


df = pd.read_csv("./data/titanic.csv")



ids_ruleset = mine_IDS_ruleset(df, rule_cutoff=40)

quant_dataframe = QuantitativeDataFrame(df)



coordinate_ascent = CoordinateAscentOptimizer(IDS(), debug=True, maximum_delta_between_iterations=200, maximum_score_estimation_iterations=3, ternary_search_precision=20)
coordinate_ascent.fit(ids_ruleset, quant_dataframe, quant_dataframe)

