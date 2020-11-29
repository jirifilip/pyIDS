from pyids.algorithms import mine_IDS_ruleset, mine_CARs
from pyids.algorithms.ids_objective_function import (
    IDSObjectiveFunction,
    ObjectiveFunctionParameters
)
from pyids.algorithms.ids import IDS

from pyarc.qcba.data_structures import QuantitativeDataFrame

import pandas as pd
import time
import numpy as np
import random
from pyarc import CBA
from pyarc.qcba import QCBA
from pyarc import TransactionDB
from pyarc.algorithms import (
    top_rules,
    createCARs,
    M1Algorithm,
    M2Algorithm
)


df_raw = pd.read_csv("c:/code/python/machine_learning/assoc_rules/train/lymph0.csv")
df_all = pd.concat(30 * [df_raw])

benchmark_data = []

time_estimation_iterations = 10
max_rules = 100
rule_cutoff = 30

def generate_lambda_array():
    lambdas = [ generate_lambda_parameter() for i in range(7) ]

    return lambdas

def generate_lambda_parameter():
    return random.randint(0, 1000)

for i in range(100, 3000, 100):
    df = df_all.iloc[:i, :].copy()
    data_count = df.shape[0]

    quant_df = QuantitativeDataFrame(df)


    for algorithm in ["DLS", "SLS", "DUSM", "RUSM"]:
        durations = []

        for _ in range(time_estimation_iterations):
            print(_)
            lambda_array = generate_lambda_array()

            print(f"data count: {data_count}")
            print(f"algorithm: {algorithm}")
            print(f"using lambda: {lambda_array}")

            cars = mine_CARs(df, rule_cutoff=rule_cutoff)

            ids = IDS(algorithm=algorithm)
            start = time.time()
            ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=lambda_array)
            duration = time.time() - start

            print(f"avg duration: {duration}")

            durations.append(duration)


        duration = np.mean(durations)

        print(f"avg duration: {duration}")

        benchmark_data.append(dict(
            data_count=data_count,
            algorithm=algorithm,
            duration=duration
        ))

    cars = mine_CARs(df, rule_cutoff=rule_cutoff)
    txns = TransactionDB.from_DataFrame(df)

    start = time.time()
    classifier = M1Algorithm(cars, txns).build()
    duration = time.time() - start

    benchmark_data.append(dict(
        data_count=data_count,
        algorithm="pyARC - M1",
        duration=duration
    ))

    start = time.time()
    classifier = M2Algorithm(cars, txns).build()
    duration = time.time() - start

    benchmark_data.append(dict(
        data_count=data_count,
        algorithm="pyARC - M2",
        duration=duration
    ))

    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df.to_csv("output_data/benchmark_datasize_sensitivity_lymph.csv", index=False)




print(benchmark_data)

