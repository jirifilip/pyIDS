from pyids.model_selection.grid_search import GridSearch
from pyids.algorithms.ids import IDS
from pyids.algorithms import mine_CARs

from pyarc.qcba.data_structures import QuantitativeDataFrame

import pandas as pd
import numpy as np

df_iris = pd.read_csv("data/iris0.csv")
quant_df = QuantitativeDataFrame(df_iris)
cars = mine_CARs(df_iris, 20)


def is_solution_interpretable(metrics):
    print(metrics)
    return (
        metrics["fraction_overlap"] <= 0.3 and
        metrics["fraction_classes"] > 1.0 and
        metrics["fraction_uncovered"] <= 0.3 and
        metrics["average_rule_width"] < 8 and
        metrics["ruleset_length"] <= 10
    )

def solution_interpretability_distance(metrics):
    distance_vector = np.array([
        max(metrics["fraction_overlap"] - 0.1, 0),
        max(1 - metrics["fraction_classes"], 0),
        max(metrics["fraction_uncovered"] - 0.15, 0),
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

    if not is_solution_interpretable(metrics):
        return 0

    auc = ids.score_auc(quant_df)

    print(auc)

    return auc



grid_search = GridSearch(
    func=fmax,
    func_args_spaces=dict(
        l1=np.linspace(1, 1000, 20),
        l2=np.linspace(1, 1000, 20),
        l3=np.linspace(1, 1000, 20),
        l4=np.linspace(1, 1000, 20),
        l5=np.linspace(1, 1000, 20),
        l6=np.linspace(1, 1000, 20),
        l7=np.linspace(1, 1000, 20)
    ),
    max_iterations=100000
)

best_lambda_parameters = grid_search.fit()

#df = pd.DataFrame(coord_asc.procedure_data)

#df.to_csv("output_data/grid_search_run_AUC_interpretability.csv")
