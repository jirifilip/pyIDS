from pyids.rule_mining import RuleMiner
import pandas as pd

from pyids.ids_classifier import IDS
from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyids.ids_ruleset import IDSRuleSet

rm = RuleMiner()

df = pd.read_csv("./data/titanic.csv")


cars = rm.mine_rules(df, minsup=0.001)
print(len(cars))

quant_df = QuantitativeDataFrame(df)

ids = IDS()
ids.fit(quant_df, cars, debug=False)

metrics = ids.score_interpretable_metrics(quant_df)

print(metrics)