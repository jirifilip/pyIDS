import pandas as pd

from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import ClassAssocationRule
from pyids.algorithms.classifier import mine_CARs
from pyids import IDS

import numpy as np


replications_n = 50
cars_to_mine = 10

df = pd.read_csv("../../../data/iris0.csv")
quant_df = QuantitativeDataFrame(df)

mined_cars_mupliple = []
mined_cars_comparison_results = []

def all_rules_same(cars1, cars2):
    for idx in range(len(cars1)):
        if repr(cars1[idx]) != repr(cars2[idx]):
            return False
        
    return True


for _ in range(replications_n):
    ClassAssocationRule.id = 0
    cars = mine_CARs(df, cars_to_mine)
    mined_cars_mupliple.append(cars)


for idx in range(replications_n):
    same = all_rules_same(mined_cars_mupliple[0], mined_cars_mupliple[idx])
    mined_cars_comparison_results.append(same)

print(np.all(mined_cars_comparison_results))
print(mined_cars_comparison_results)

ids_models_multiple = []
ids_comparison_results = []

for _ in range(replications_n):
    ids = IDS()
    ids = ids.fit(dataframe=quant_df, rules=cars, debug=False, random_seed=2)
    ids_models_multiple.append(ids.clf.rules)


for idx in range(replications_n):
    same = all_rules_same(ids_models_multiple[0], ids_models_multiple[idx])
    ids_comparison_results.append(same)


print(np.all(ids_comparison_results))
print(ids_comparison_results)