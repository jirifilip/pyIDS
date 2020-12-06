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
    "anneal": 20,

}

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
    return [x for x in os.listdir(path) if x.startswith(dataset_name)]


data_path = "C:/code/python/machine_learning/assoc_rules/"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "../../../test")


def get_dataset_files(path, dataset_name):
    return [x for x in os.listdir(path) if x.startswith(dataset_name)]


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


        def fmax(param_dict):
            print(param_dict)

            support, confidence = param_dict["support"] / 1000, param_dict["confidence"] / 1000
            print(dict(support=support, confidence=confidence))

            cba = CBA(support=support, confidence=confidence)
            cba.fit(txns)

            cba_clf = cba.clf

            ids = IDS()
            ids_clf = IDSClassifier(IDSRuleSet.from_cba_rules(cba_clf.rules).ruleset)
            ids_clf.quant_dataframe_train = quant_df
            ids_clf.calculate_default_class()

            ids.clf = ids_clf

            metrics = ids.score_interpretability_metrics(quant_df_test)
            if not is_solution_interpretable(metrics):
                print(metrics)
                return 0

            auc = ids.score_auc(quant_df_test)

            print(auc)

            return auc


        coord_asc = CoordinateAscent(
            func=fmax,
            func_args_ranges=dict(
                support=(150, 999),
                confidence=(150, 999),
            ),
            ternary_search_precision=5,
            max_iterations=2,
            extension_precision=-1,
            func_args_extension=dict(
                support=0,
                confidence=0
            )
        )

        best_pars = coord_asc.fit()

        print("best_pars:", best_pars)
        support, confidence = best_pars[0] / 1000, best_pars[1] / 1000


        cba = CBA(support=support, confidence=confidence)
        cba.fit(txns)
        cba_clf = cba.clf

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
        benchmark_data_df.to_csv("output_data/cba_auc_interpretability=0_coats_benchmark-anneal.csv")


