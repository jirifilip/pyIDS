import pandas as pd
from pyids.ids_classifier import IDSOneVsAll, mine_IDS_ruleset
from pyids.model_selection import CoordinateAscentOptimizer, train_test_split_pd

from pyarc.qcba.data_structures import QuantitativeDataFrame


df = pd.read_csv("./data/iris0.csv")
df_train, df_test = train_test_split_pd(df, prop=0.2)

ids_ruleset = mine_IDS_ruleset(df_train, rule_cutoff=50)

quant_dataframe_train = QuantitativeDataFrame(df_train)
quant_dataframe_test = QuantitativeDataFrame(df_test)

coordinate_ascent = CoordinateAscentOptimizer(IDSOneVsAll(), debug=True, maximum_delta_between_iterations=200, maximum_score_estimation_iterations=3)
coordinate_ascent.fit(ids_ruleset, quant_dataframe_train, quant_dataframe_test)

best_lambda_array = coordinate_ascent.current_best_params

