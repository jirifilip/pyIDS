import pandas as pd
import pyfpgrowth
from pyfpgrowth.pyfpgrowth import FPTree

from pyarc.data_structures import TransactionDB

df = pd.read_csv("../../../data/iris0.csv")
txns = TransactionDB.from_DataFrame(df)

print(txns.string_representation)


fp_tree = FPTree(txns.string_representation, 0.01, None, None)

print(fp_tree.headers)