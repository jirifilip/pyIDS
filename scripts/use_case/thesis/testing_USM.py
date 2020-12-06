import pandas as pd
import numpy as np
from pyids.algorithms.ids import IDS
from pyids.algorithms import mine_CARs

from pyarc.qcba.data_structures import QuantitativeDataFrame

import random 
import time


df = pd.read_csv("../../../data/iris0.csv")
cars = mine_CARs(df, 15, sample=False)

quant_dataframe = QuantitativeDataFrame(df)

start = time.time()
ids = IDS(algorithm="RUSM")
ids.fit(class_association_rules=cars, quant_dataframe=quant_dataframe, random_seed=None, lambda_array=[1, 1, 1, 1, 1, 1, 1000000000])
end = time.time()

print(end - start)

for r in ids.clf.rules:
    print(r)

auc = ids.score_auc(quant_dataframe)

print("AUC", auc)