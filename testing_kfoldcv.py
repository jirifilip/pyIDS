from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import scipy

import numpy as np
import pandas as pd

from pyarc.qcba import *

from pyarc.algorithms import createCARs, top_rules
from pyarc import TransactionDB

from pyids import IDS
from pyids.ids_cacher import IDSCacher
from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_rule import IDSRule
from pyids.ids_classifier import IDSOneVsAll
from pyids.model_selection import encode_label, mode, KFoldCV


dataframes = [ pd.read_csv("./data/iris{}.csv".format(i)) for i in range(10)]

kfold = KFoldCV(IDSOneVsAll(), dataframes, score_auc=True)
scores = kfold.fit(50)

print(scores)