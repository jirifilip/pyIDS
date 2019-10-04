import pandas as pd
import numpy as np
from pyids.data_structures import IDS, mine_CARs

from pyarc.qcba.data_structures import QuantitativeDataFrame

import random 


df = pd.read_csv("./data/titanic.csv")
cars = mine_CARs(df, 40)

quant_dataframe = QuantitativeDataFrame(df)


ids = IDS()
ids.fit(class_association_rules=cars, quant_dataframe=quant_dataframe, debug=False, random_seed=12)

auc = ids.score_auc(quant_dataframe)
print(auc)


for r in ids.clf.rules:
    print(r)