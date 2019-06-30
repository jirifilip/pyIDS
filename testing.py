import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.metrics import accuracy_score

from pyids.ids_rule import IDSRule
from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_objective_function import ObjectiveFunctionParameters, IDSObjectiveFunction
from pyids.ids_optimizer import RSOptimizer, SLSOptimizer
from pyids.ids_cacher import IDSCacher
from pyids.ids_classifier import IDS

from pyarc.qcba import *

import cProfile




from pyarc.algorithms import createCARs, top_rules
from pyarc import TransactionDB


df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/iris0.csv")
Y = df.iloc[:,-1]
txns = TransactionDB.from_DataFrame(df)
rules = top_rules(txns.string_representation, appearance=txns.appeardict)
cars = createCARs(rules)


quant_df = QuantitativeDataFrame(df)

"""
car1 = cars[0]
car2 = cars[10]

ids_rules = list(map(IDSRule, cars))

ids_rule1 = IDSRule(car1)
ids_rule2 = IDSRule(car2)


params = ObjectiveFunctionParameters()
all_rules = IDSRuleSet(ids_rules[:80])
params.params["all_rules"] = all_rules
params.params["quant_dataframe"] = quant_df
objective_function = IDSObjectiveFunction(objective_func_params=params)


solution_set = IDSRuleSet([ids_rule1, ids_rule2])




opt = SLSOptimizer(objective_function, params)

#opt.optimize()

rs_opt = RSOptimizer(all_rules.ruleset)

soln_set = IDSRuleSet(rs_opt.optimize())


print(objective_function.evaluate(soln_set))

sampled = opt.sample_random_set(soln_set.ruleset, 0.33)

objective_function.evaluate(IDSRuleSet(sampled))

soln_set = opt.optimize()


print(soln_set)
"""

ids = IDS()
ids.fit(quant_df, cars[:40])

acc = ids.score(quant_df)

print(acc)