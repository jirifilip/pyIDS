from pyids.rule_mining import RuleMiner
import pandas as pd

from pyids import IDS
from pyarc.qcba.data_structures import QuantitativeDataFrame

rm = RuleMiner()

df = pd.read_csv("./data/titanic.csv")


cars = rm.mine_rules(df, minsup=0.001)
print(len(cars))
