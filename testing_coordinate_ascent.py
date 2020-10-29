import numpy as np
import pandas as pd
from pyarc.qcba import *

from pyarc.algorithms import createCARs, top_rules
from pyarc import TransactionDB


from pyids.algorithms.ids_classifier import mine_IDS_ruleset
from pyids.algorithms.ids_multiclass import IDSOneVsAll
from pyids.algorithms.ids import IDS
from pyids.model_selection import encode_label, mode, CoordinateAscentOptimizer


df = pd.read_csv("./data/iris0.csv")



ids_ruleset = mine_IDS_ruleset(df, rule_cutoff=40)

quant_dataframe = QuantitativeDataFrame(df)



coordinate_ascent = CoordinateAscentOptimizer(
    IDSOneVsAll(algorithm="SLS"),
    maximum_delta_between_iterations=200,
    maximum_score_estimation_iterations=1,
    ternary_search_precision=20
)
coordinate_ascent.fit(ids_ruleset, quant_dataframe, quant_dataframe)

