import pandas as pd

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_classifier import IDS, mine_CARs
from pyids.model_selection import CoordinateAscentOptimizer, train_test_split_pd, calculate_ruleset_statistics


df = pd.read_csv("../../data/titanic.csv")



cars = mine_CARs(df, rule_cutoff=40)

df_train, df_test = train_test_split_pd(df, prop=0.25)
quant_df_train, quant_df_test = QuantitativeDataFrame(df_train), QuantitativeDataFrame(df_test)


lambda_array_coordinate_ascent = [18, 18, 18, 18, 18, 18, 18]
lambda_array_random_search = [510.6938775510204, 694.1836734693878, 918.4489795918367, 286.42857142857144, 490.30612244897964, 694.1836734693878, 62.163265306122454]


ids_ascent = IDS()
ids_random = IDS()
ids_basic = IDS()

ids_ascent.fit(quant_df_train, cars, debug=False, lambda_array=lambda_array_coordinate_ascent)
ids_random.fit(quant_df_train, cars, debug=False, lambda_array=lambda_array_random_search)
ids_basic.fit(quant_df_train, cars, debug=False, lambda_array=7*[1])

ascent_metrics = calculate_ruleset_statistics(IDSRuleSet(ids_ascent.clf.rules), quant_df_test)
random_metrics = calculate_ruleset_statistics(IDSRuleSet(ids_random.clf.rules), quant_df_test)
basic_metrics = calculate_ruleset_statistics(IDSRuleSet(ids_basic.clf.rules), quant_df_test)

ascent_metrics.update({"auc": ids_ascent.score_auc(quant_df_test)})
random_metrics.update({"auc": ids_random.score_auc(quant_df_test)})
basic_metrics.update({"auc": ids_basic.score_auc(quant_df_test)})


metrics_dict = dict(
    ascent=ascent_metrics,
    random=random_metrics,
    basic=basic_metrics
)

metrics_df = pd.DataFrame(metrics_dict)


metrics_df.to_latex("results/metrics_table.latex")