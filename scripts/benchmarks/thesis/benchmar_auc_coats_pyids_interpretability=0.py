import os
import pandas as pd
import numpy as np

from pyids.algorithms.ids import IDS
from pyids.algorithms import mine_CARs
from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyids.model_selection.coordinate_ascent import CoordinateAscent

dataset_rulenum = {
    "anneal": 15,
    "iris": 20,
    "lymph": 20
}

data_path = "C:/code/python/machine_learning/assoc_rules/"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "../../../test")

interpretability_bounds = dict(
    fraction_overlap=0.1,
    fraction_classes=1,
    fraction_uncovered=0.15,
    average_rule_width=8,
    ruleset_length=10
)

def is_solution_interpretable(metrics):
    print(metrics)
    return (
        metrics["fraction_overlap"] <= interpretability_bounds["fraction_overlap"] and
        metrics["fraction_classes"] >= interpretability_bounds["fraction_classes"] and
        metrics["fraction_uncovered"] <= interpretability_bounds["fraction_uncovered"] and
        metrics["average_rule_width"] <= interpretability_bounds["average_rule_width"] and
        metrics["ruleset_length"] <= interpretability_bounds["ruleset_length"]
    )

def solution_interpretability_distance(metrics):
    distance_vector = np.array([
        max(metrics["fraction_overlap"] - interpretability_bounds["fraction_overlap"], 0),
        max(interpretability_bounds["fraction_classes"] - metrics["fraction_classes"], 0),
        max(metrics["fraction_uncovered"] - interpretability_bounds["fraction_uncovered"], 0),
        max(metrics["average_rule_width"] - interpretability_bounds["average_rule_width"], 0),
        max(metrics["ruleset_length"] - interpretability_bounds["ruleset_length"], 0)
    ])
    return np.sum(distance_vector)


def get_dataset_files(path, dataset_name):
    return [ x for x in os.listdir(path) if x.startswith(dataset_name) ]


benchmark_data = []


for dataset_name, rule_cutoff in dataset_rulenum.items():
    print(dataset_name)

    train_files = get_dataset_files(train_path, dataset_name)
    test_files = get_dataset_files(test_path, dataset_name)
    for algorithm in ["RUSM", "DUSM", "SLS", "DLS"]:
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
                ids.fit(rules=cars, dataframe=quant_df, lambda_array=list(lambda_dict.values()))

                metrics = ids.score_interpretability_metrics(quant_df)

                if not is_solution_interpretable(metrics):

                    print(0)

                    return 0

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




            scores = []

            for _ in range(10):
                ids = IDS(algorithm=algorithm)
                ids.fit(quant_df, cars, lambda_array=best_lambda)
                auc = ids.score_auc(quant_df_test)
                metrics = ids.score_interpretability_metrics(quant_df_test)

                interpretable = is_solution_interpretable(metrics)

                scores.append(dict(
                    auc=auc,
                    metrics=metrics,
                    interpretable=interpretable
                ))

            scores.sort(key=lambda x: (x["interpretable"], x["auc"]), reverse=True)

            data = dict(
                dataset_name=dataset_name,
                algorithm=algorithm,
                auc=scores[0]["auc"],
                rule_cutoff=rule_cutoff
            )

            data.update(scores[0]["metrics"])

            print(data)

            benchmark_data.append(data)

            benchmark_data_df = pd.DataFrame(benchmark_data)
            benchmark_data_df.to_csv("output_data/auc_coats_interpretability=0_pyids_benchmark_all.csv")


