import pandas as pd

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.ids_classifier import IDS, mine_IDS_ruleset
from pyids.model_selection import CoordinateAscentOptimizer, train_test_split_pd


df = pd.read_csv("../../data/titanic.csv")



ids_ruleset = mine_IDS_ruleset(df, rule_cutoff=40)

df_train, df_test = train_test_split_pd(df, prop=0.25)
quant_df_train, quant_df_test = QuantitativeDataFrame(df_train), QuantitativeDataFrame(df_test)



coordinate_ascent = CoordinateAscentOptimizer(IDS(), maximum_delta_between_iterations=200, maximum_score_estimation_iterations=10, ternary_search_precision=20, maximum_consecutive_iterations=20)
lambda_array = coordinate_ascent.fit(ids_ruleset, quant_df_train, quant_df_test)

print(lambda_array)

with open("results/coordinate_ascent_lambda_array.txt", "w") as file:
    file.write(str(lambda_array))

