import pandas as pd

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.ids_classifier import IDS, mine_IDS_ruleset
from pyids.model_selection import RandomSearchOptimizer, train_test_split_pd


df = pd.read_csv("../../data/titanic.csv")



ids_ruleset = mine_IDS_ruleset(df, rule_cutoff=40)

df_train, df_test = train_test_split_pd(df, prop=0.25)
quant_df_train, quant_df_test = QuantitativeDataFrame(df_train), QuantitativeDataFrame(df_test)



random_optimizer = RandomSearchOptimizer(IDS(), maximum_score_estimation_iterations=5, maximum_iterations=500)
lambda_array = random_optimizer.fit(ids_ruleset, quant_df_train, quant_df_test)
all_params = random_optimizer.score_params_dict

print(lambda_array)

with open("results/random_search_lambda_array.txt", "w") as file:
    file.write(str(lambda_array))

with open("results/random_search_all_score_params.txt", "w") as file:
    file.write(str(random_optimizer.score_params_dict))

