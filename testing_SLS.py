import pandas as pd
import numpy as np
from pyids.data_structures import IDS, mine_CARs

from pyarc.qcba.data_structures import QuantitativeDataFrame

import random 


df = pd.read_csv("./data/titanic.csv")


cars = mine_CARs(df, rule_cutoff=10, target_rule_count=1000, conf_step=0.20, random_seed=5)
for r in cars:
    print(r)

print("=======")

quant_dataframe = QuantitativeDataFrame(df)


ids = IDS()
ids.fit(class_association_rules=cars, quant_dataframe=quant_dataframe, debug=False, random_seed=12)

auc = ids.score_auc(quant_dataframe)
print(auc)

for r in ids.clf.rules:
    print(r)