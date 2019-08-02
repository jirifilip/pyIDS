import pandas as pd
from collections import Counter

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.ids_classifier import IDS, mine_CARs
df = pd.read_csv("./data/titanic.csv")


cars = mine_CARs(df, 20)

quant_dataframe = QuantitativeDataFrame(df)




ids = IDS()

ids.fit(quant_dataframe, class_association_rules=cars, debug=False)

print(ids.score_auc(quant_dataframe, confidence_based=False))
print(ids.score_auc(quant_dataframe, confidence_based=True))
