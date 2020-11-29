import os
import pandas as pd
import numpy as np

from pyids.algorithms.ids import IDS
from pyids.algorithms import mine_CARs
from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyids.model_selection.coordinate_ascent import CoordinateAscent

lambda_dict = {'l1': 124.16415180612711, 'l2': 38.896662094192955, 'l3': 557.0996799268405, 'l4': 638.188385916781, 'l5': 136.48056698673983, 'l6': 432.1760402377687, 'l7': 452.1563786008231}
lambda_array = [665.9341563786008, 271.7242798353909, 212.34156378600824, 20.489711934156375, 648.5761316872428, 911, 560]



df = pd.read_csv("C:/code/python/machine_learning/assoc_rules/train/iris0.csv")
quant_df = QuantitativeDataFrame(df)
quant_df_test = QuantitativeDataFrame(pd.read_csv("C:/code/python/machine_learning/assoc_rules/test/iris0.csv"))

cars = mine_CARs(df, 20)

ids = IDS(algorithm="DUSM")
ids.fit(quant_df, cars, lambda_array=lambda_array)

print(ids.score_auc(quant_df))
