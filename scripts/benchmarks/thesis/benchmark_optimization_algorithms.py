from pyids.algorithms import mine_IDS_ruleset, mine_CARs
from pyids.algorithms.objective_function import (
    IDSObjectiveFunction,
    ObjectiveFunctionParameters
)
from pyids.algorithms.ids import IDS

from pyarc.qcba.data_structures import QuantitativeDataFrame

import pandas as pd
import time
import numpy as np
import random


df = pd.read_csv("../../../data/iris0.csv")
quant_df = QuantitativeDataFrame(df)

benchmark_data = []

def generate_lambda_array():
    lambdas = [ generate_lambda_parameter() for i in range(7) ]

    return lambdas

def generate_lambda_parameter():
    return random.randint(0, 1000)

for rule_num in range(5, 105, 5):
    for algorithm in ["DLS", "SLS", "DUSM", "RUSM"]:
        durations = []

        for _ in range(10):
            print(_)
            lambda_array = generate_lambda_array()

            print(f"rule num: {rule_num}")
            print(f"algorithm: {algorithm}")
            print(f"using lambda: {lambda_array}")

            cars = mine_CARs(df, rule_cutoff=rule_num)

            ids = IDS(algorithm=algorithm)
            start = time.time()
            ids.fit(rules=cars, dataframe=quant_df, lambda_array=lambda_array)
            duration = time.time() - start

            print(f"duration: {duration}")

            durations.append(duration)


        duration = np.mean(durations)

        print(f"avg duration: {duration}")

        benchmark_data.append(dict(
            rule_num=rule_num,
            algorithm=algorithm,
            duration=duration
        ))

    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv("output_data/benchmark_optimization_algorithms.csv", index=False)



print(benchmark_data)

