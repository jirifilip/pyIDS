import pandas as pd
import numpy as np
from pyids.data_structures import IDS, mine_CARs, IDSRuleSet

from pyarc.qcba.data_structures import QuantitativeDataFrame

import random 



df = pd.read_csv("./data/titanic.csv")
cars = mine_CARs(df, 40, sample=True)
ids_ruleset = IDSRuleSet.from_cba_rules(cars).ruleset

quant_dataframe = QuantitativeDataFrame(df)

for r in reversed(sorted(cars)):
    print(r)



ids = IDS()
ids.fit(class_association_rules=cars, quant_dataframe=quant_dataframe, debug=True, random_seed=None, lambda_array=[0, 0, 0, 0, 0, 1, 0])

for r in ids.clf.rules:
    print(r)

auc = ids.score_auc(quant_dataframe)
print(auc)
print(ids.score_interpretable_metrics(quant_dataframe))



print()
print()
r0 = list(ids_ruleset)[15]
print(r0)
print(np.sum(r0._correct_cover(quant_dataframe)))

