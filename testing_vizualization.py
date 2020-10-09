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
from pyids.ids_classifier import IDS, mine_CARs

from pyids.visualization import IDSVisualization



from pyarc.qcba import *


df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/iris0.csv")
cars = mine_CARs(df, 20)


quant_df = QuantitativeDataFrame(df)

ids = IDS()
ids.fit(quant_df, cars, debug=True)


viz = IDSVisualization(ids, quant_df)

viz.visualize_dataframe()