import pandas as pd
import numpy as np
from pyids.data_structures import IDS, mine_CARs, IDSRuleSet

from pyarc.qcba.data_structures import QuantitativeDataFrame

import random 
import time


df = pd.read_csv("./data/titanic.csv")
cars = mine_CARs(df, 15, sample=False)
ids_ruleset = IDSRuleSet.from_cba_rules(cars).ruleset

quant_dataframe = QuantitativeDataFrame(df)

for r in reversed(sorted(cars)):
    print(r)


start = time.time()
ids = IDS()
ids.fit(class_association_rules=cars, quant_dataframe=quant_dataframe, debug=False, random_seed=None, lambda_array=[1, 0, 0, 0, 0, 0, 0])
end = time.time()

print(end - start)

for r in ids.clf.rules:
    print(r)

auc = ids.score_auc(quant_dataframe)