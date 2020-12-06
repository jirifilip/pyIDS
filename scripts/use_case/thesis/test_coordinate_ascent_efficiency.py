from pyids.model_selection import CoordinateAscent
from pyids.algorithms.ids import IDS
from pyids.algorithms import mine_CARs, mine_IDS_ruleset

from pyarc.qcba.data_structures import QuantitativeDataFrame

import pandas as pd
import numpy as np

df_iris = pd.read_csv("../../../data/iris0.csv")
quant_df = QuantitativeDataFrame(df_iris)
cars = mine_CARs(df_iris, 20)


def is_solution_interpretable(metrics):
    print(metrics)
    return (
        metrics["fraction_overlap"] <= 0.5 and
        metrics["fraction_classes"] > 1.0 and
        metrics["fraction_uncovered"] <= 0.5 and
        metrics["average_rule_width"] < 8 and
        metrics["ruleset_length"] <= 10
    )

def solution_interpretability_distance(metrics):
    distance_vector = np.array([
        max(metrics["fraction_overlap"] - 0.5, 0),
        max(1 - metrics["fraction_classes"], 0),
        max(metrics["fraction_uncovered"] - 0.5, 0),
        max(metrics["average_rule_width"] - 8, 0),
        max(metrics["ruleset_length"] - 10, 0)
    ])
    return np.sum(distance_vector)
    return np.linalg.norm(distance_vector)


def fmax(lambda_dict):
    print(lambda_dict)
    ids = IDS(algorithm="SLS")
    ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=list(lambda_dict.values()))

    metrics = ids.score_interpretability_metrics(quant_df)

    """
    if not is_solution_interpretable(metrics):
        distance = -solution_interpretability_distance(metrics)
        print(distance)
        return -distance
    """

    auc = ids.score_auc(quant_df)

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
    max_iterations=3
)

coord_asc.fit()

df = pd.DataFrame(coord_asc.procedure_data)

df.to_csv("output_data/coordinate_ascent_run_AUConly.csv")
