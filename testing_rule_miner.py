from pyids.rule_mining import RuleMiner
import pandas as pd

from pyids.ids_classifier import IDS
from pyarc.qcba.data_structures import QuantitativeDataFrame

rm = RuleMiner()

df = pd.read_csv("./data/iris0.csv")


cars = rm.mine_rules(df)

cars = cars[:40]

quant_df = QuantitativeDataFrame(df)

ids = IDS()
ids.fit(quant_df, cars, debug=False)

metrics = ids.score_interpretable_metrics(quant_df)

print(metrics)