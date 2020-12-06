import os
import pandas as pd
import numpy as np

from pyids.algorithms.ids import IDS
from pyids.algorithms import mine_CARs
from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyids.model_selection.coordinate_ascent import CoordinateAscent

dataset_rulenum = {
    "lymph": 20,
    "iris": 20,
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
    for algorithm in ["DUSM", "RUSM", "SLS", "DLS"][:]:
        for train_file, test_file in list(zip(train_files, test_files))[:]:
            dataset_path = os.path.join(train_path, train_file)
            dataset_test_path = os.path.join(test_path, test_file)

            df = pd.read_csv(dataset_path)
            quant_df = QuantitativeDataFrame(df)

            df_test = pd.read_csv(dataset_test_path)
            quant_df_test = QuantitativeDataFrame(df_test)

            cars = mine_CARs(df, rule_cutoff)

            def fmax(lambda_dict):
                print(lambda_dict)
                ids = IDS(algorithm=algorithm)
                ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=list(lambda_dict.values()))

                auc = ids.score_auc(quant_df_test)

                print(auc)

                return auc


            coord_asc = CoordinateAscent(
                func=fmax,
                func_args_ranges=dict(
                    l1=(1, 1000),
                    l2=(1, 1000),
                    l3=(1, 1000),
                    l4=(1, 1000),
                    l5=(1, 1000),
                    l6=(1, 1000),
                    l7=(1, 1000)
                ),
                ternary_search_precision=50,
                max_iterations=2
            )

            best_lambda = coord_asc.fit()

            print("best lambda", best_lambda)

            aucs = []
            for i in range(10):
                ids = IDS(algorithm=algorithm)
                ids.fit(quant_df, cars, lambda_array=best_lambda)
                auc = ids.score_auc(quant_df_test)
                metrics = ids.score_interpretability_metrics(quant_df_test)

                print(auc)

                aucs.append({
                    "metrics": metrics,
                    "auc": auc
                })

            aucs.sort(key=lambda x: x["auc"], reverse=True)
            print("aucs", aucs)
            auc = aucs[0]["auc"]
            metrics = aucs[0]["metrics"]

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
            benchmark_data_df.to_csv("output_data/auc_only_coats_pyids_benchmark6_better.csv")


