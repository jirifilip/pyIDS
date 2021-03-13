import pandas as pd
import numpy as np
from pyids import IDS
from pyids.algorithms import mine_CARs
from pyids.data_structures import IDSRuleSet
import json
from pyarc.qcba.data_structures import QuantitativeDataFrame

import random
import logging
import time

logging.basicConfig(level=logging.INFO)

df = pd.read_csv("../../../data/iris0.csv")
cars = mine_CARs(df, 10, sample=False)
ids_ruleset = IDSRuleSet.from_cba_rules(cars).ruleset

quant_dataframe = QuantitativeDataFrame(df)


def generate_lambda_arr(one_idx):
    total_params = 7

    if one_idx == 0:
        start_arr = []
    else:
        start_arr = one_idx * [0]

    end_arr = (total_params - one_idx - 1) * [0]

    return start_arr + [1] + end_arr


criteria_time = {}
criteria_time_default = {}
for i in range(7):
    criteria_time[i] = []
    criteria_time_default[i] = []
    lambda_array = generate_lambda_arr(i)

    for _ in range(1):
        start = time.time()
        ids = IDS(algorithm="SLS")
        ids.fit(
            rules=cars,
            dataframe=quant_dataframe,
            random_seed=None,
            lambda_array=lambda_array,
            optimizer_args=(dict(
                max_omega_iterations=10
            ))
        )
        end = time.time()

        duration = end - start
        criteria_time_default[i].append(duration)

        print(f"f{i + 1} default: {duration}")

        start = time.time()
        ids = IDS(algorithm="SLS")
        ids.fit(
            rules=cars,
            dataframe=quant_dataframe,
            random_seed=None,
            lambda_array=lambda_array,
            optimizer_args=(dict(
                max_omega_iterations=10000
            ))
        )
        end = time.time()

        duration = end - start
        criteria_time[i].append(duration)

        print(f"f{i + 1} 10000: {duration}")


print(criteria_time)
print(criteria_time_default)

output = dict(better=criteria_time, default=criteria_time_default)

json.dump(output, open("../../../output_data/benchmark_f_se.json", "w"))