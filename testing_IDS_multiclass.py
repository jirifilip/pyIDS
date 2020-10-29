import pandas as pd
import numpy as np
from pyids import IDS
from pyids.algorithms import mine_CARs
from pyids.algorithms.ids_multiclass import IDSOneVsAll
from pyids.data_structures import IDSRuleSet
from pyarc.qcba.data_structures import QuantitativeDataFrame
import random
import logging
import time

logging.basicConfig(level=logging.DEBUG)

df = pd.read_csv("./data/iris0.csv")
quant_dataframe = QuantitativeDataFrame(df)

ids_multiclass = IDSOneVsAll(algorithm="DUSM")
ids_multiclass.fit(quant_dataframe, lambda_array=[1, 1, 0, 0, 100000, 10000, 1000000], rule_cutoff=30)

auc = ids_multiclass.score_auc(quant_dataframe)

print(auc)

summary_df = ids_multiclass.summary()
print(summary_df.to_string())

print(ids_multiclass.score_interpretability_metrics(quant_dataframe))