from astropy.nddata import support_nddata
from pyarc import CBA
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import M1Algorithm

import os
import pandas as pd
import numpy as np

from pyids.algorithms.ids import IDS
from pyids.algorithms.ids_classifier import IDSClassifier
from pyids.data_structures import IDSRuleSet, IDSRule

from pyids.algorithms import mine_CARs
from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyids.model_selection.coordinate_ascent import CoordinateAscent

dataset_rulenum = {
    "iris": 20,
    "lymph": 20,
    "anneal": 20,
    "sick": 20,
    "glass": 20,
    "balance-scale": 20
}

data_path = "C:/code/python/machine_learning/assoc_rules/"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "../../../test")


def get_dataset_files(path, dataset_name):
    return [ x for x in os.listdir(path) if x.startswith(dataset_name) ]


benchmark_data = []


for dataset_name, rule_cutoff in dataset_rulenum.items():
    print(dataset_name)

    train_files = get_dataset_files(train_path, dataset_name)
    test_files = get_dataset_files(test_path, dataset_name)
    for train_file, test_file in list(zip(train_files, test_files))[:]:
        dataset_path = os.path.join(train_path, train_file)
        dataset_test_path = os.path.join(test_path, test_file)

        df = pd.read_csv(dataset_path)
        quant_df = QuantitativeDataFrame(df)
        txns = TransactionDB.from_DataFrame(df)

        df_test = pd.read_csv(dataset_test_path)
        quant_df_test = QuantitativeDataFrame(df_test)
        txns_test = TransactionDB.from_DataFrame(df_test)

        cars = mine_CARs(df, rule_cutoff)

        cba_clf = M1Algorithm(dataset=txns, rules=cars).build()

        ids = IDS()
        ids_clf = IDSClassifier(IDSRuleSet.from_cba_rules(cba_clf.rules).ruleset)
        ids_clf.quant_dataframe_train = quant_df
        ids_clf.calculate_default_class()

        ids.clf = ids_clf




        data = dict(
            dataset_name=dataset_name,
            algorithm="pyARC",
            auc=ids.score_auc(quant_df_test, order_type="cba"),
            rule_cutoff=rule_cutoff
        )

        data.update(ids.score_interpretability_metrics(quant_df_test))

        print(data)

        benchmark_data.append(data)

        benchmark_data_df = pd.DataFrame(benchmark_data)
        benchmark_data_df.to_csv("output_data/cba_auc_interpretability_benchmark.csv")


