import pandas as pd
import numpy as np

from pyarc.data_structures import TransactionDB
from pyarc.algorithms import top_rules, createCARs

df = pd.read_csv("../../../data/iris0.csv")

df[df["sepallength"] == "-inf_to_5.55"] = np.NaN

print(df)

txns = TransactionDB.from_DataFrame(df)

rules = top_rules(txns.string_representation, appearance=txns.appeardict)

cars = createCARs(rules)

for car in cars[:10]:
    print(car)