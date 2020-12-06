import os
import pandas as pd
import numpy as np

from pyids.algorithms.ids import IDS
from pyids.algorithms import mine_CARs
from pyarc.qcba.data_structures import QuantitativeDataFrame

dataset_rulenum = {
    "glass": 40,
    "balance-scale": 40,
    "labor": 40,
    "zoo": 40,
    "breast-w": 40,
    "diabetes": 40,
    "car": 40,
    "anneal": 40,
    "soybean": 40,
    "vehicle": 40,
    "ionosphere": 40,
    "kr-vs-kp": 40,
    "waveform-5000": 40,
    "australian": 40,
    "credit-g": 40,
    "mushroom": 40,
    "sick": 40,
    "iris": 40,
    "lymph": 40,
    "letter": 40,
    "hypothyroid": 40,
    "vowel": 40
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
    for algorithm in ["SLS", "DLS", "DUSM", "RUSM"]:
        for train_file, test_file in zip(train_files, test_files):
            dataset_path = os.path.join(train_path, train_file)
            dataset_test_path = os.path.join(test_path, test_file)

            df = pd.read_csv(dataset_path)
            quant_df = QuantitativeDataFrame(df)

            df_test = pd.read_csv(dataset_test_path)
            quant_df_test = QuantitativeDataFrame(df_test)

            cars = mine_CARs(df, rule_cutoff)

            ids = IDS(algorithm=algorithm)
            ids.fit(quant_df, cars)

            auc = ids.score_auc(quant_df_test)
            metrics = ids.score_interpretability_metrics(quant_df_test)

            data = dict(
                dataset_name=dataset_name,
                algorithm=algorithm,
                auc=auc,
                rule_cutoff=rule_cutoff
            )

            data.update(metrics)

            print(data)

            benchmark_data.append(data)

            benchmark_data_df = pd.DataFrame(benchmark_data)
            benchmark_data_df.to_csv("output_data/auc_only_pyids_benchmark.csv")


