import pandas as pd
import numpy as np
from pyids import IDS
from pyids.algorithms import mine_CARs
from pyids.data_structures import IDSRuleSet

from pyarc.qcba.data_structures import QuantitativeDataFrame

import random
import logging
import time

logging.basicConfig(level=logging.INFO)

df = pd.read_csv("../../../data/iris0.csv")
cars = mine_CARs(df, 50, sample=False)
ids_ruleset = IDSRuleSet.from_cba_rules(cars).ruleset

quant_dataframe = QuantitativeDataFrame(df)

start = time.time()
ids = IDS(algorithm="RUSM")
ids.fit(
    class_association_rules=cars,
    quant_dataframe=quant_dataframe,
    random_seed=None,
    lambda_array=7*[1]
)
end = time.time()

print("time", end - start)

for r in ids.clf.rules:
    print(r)

#auc_cba = ids.score_auc(quant_dataframe, order_type="cba")
#auc_f1 = ids.score_auc(quant_dataframe, order_type="f1")

#print(auc_cba, auc_f1)
#print(ids.score(quant_dataframe))
print(ids.score_auc(quant_dataframe))