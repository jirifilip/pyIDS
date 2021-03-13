import os
import re
import pandas as pd
import logging
import numpy as np

from pyarc.qcba.data_structures import QuantitativeDataFrame
import time

from pyids.algorithms.classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import M1Algorithm

#logging.basicConfig(level=logging.DEBUG)

iris_file = "c:/code/python/machine_learning/assoc_rules/train/iris0.csv"

df = pd.read_csv(iris_file)
txns = TransactionDB.from_DataFrame(df)

iris_benchmark = []

for i in range(10, 110, 10):
    rule_count = i

    rules = mine_CARs(df, rule_count)

    quant_df = QuantitativeDataFrame(df)

    cars = mine_CARs(df, rule_count)
    print(len(cars))

    for algorithm in ["DLS", "SLS"]:
        times = []

        for _ in range(10):
            start_time = time.time()

            ids = IDS(algorithm=algorithm)
            ids.fit(dataframe=quant_df, rules=cars)

            total_time = time.time() - start_time
            times.append(total_time)

        benchmark_data = dict(
            rule_count=rule_count, duration=np.mean(times), algorithm=algorithm
        )

        print(benchmark_data)

        iris_benchmark.append(benchmark_data)

    times = []

    for _ in range(10):
        start_time = time.time()

        clf = M1Algorithm(cars, txns).build()

        total_time = time.time() - start_time
        times.append(total_time)

    benchmark_data = dict(
        rule_count=rule_count, duration=np.mean(times), algorithm="pyARC cba"
    )

    print(benchmark_data)

    iris_benchmark.append(benchmark_data)


benchmark_df = pd.DataFrame(iris_benchmark)
benchmark_df.to_csv("output_data/iris_speed_benchmark.csv", index=False)