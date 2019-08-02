
# coding: utf-8

# # Dataset size sensitivity benchmark

# In[29]:

RULE_COUNT_TO_USE = 15
PY_IDS_DURATION_ITERATIONS = 10


# # Guide to use lvhimabindu/interpretable_decision_sets
# 
# * git pull https://github.com/lvhimabindu/interpretable_decision_sets interpretable_decision_sets_lakkaraju
# * locate your python *site_packages* directory
# * copy *interpretable_decision_sets_lakkaraju* into *site_packages*
# * correct errors in code to allow it to run (wrong identation etc.)

# # Interpretable Decision Sets - setup

# In[3]:

import interpretable_decision_sets_lakkaraju.IDS_smooth_local as sls_lakk
from interpretable_decision_sets_lakkaraju.IDS_smooth_local import run_apriori, createrules, smooth_local_search, func_evaluation


# In[4]:

import pandas as pd
import numpy as np
import time


# ## Simple example

# In[5]:

df = pd.read_csv('../../data/titanic_train.tab',' ', header=None, names=['Passenger_Cat', 'Age_Cat', 'Gender'])
df1 = pd.read_csv('../../data/titanic_train.Y', ' ', header=None, names=['Died', 'Survived'])
Y = list(df1['Died'].values)

itemsets = run_apriori(df, 0.1)
list_of_rules = createrules(itemsets, list(set(Y)))


# In[6]:

support_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
support_levels = list(reversed(support_levels))
data_size_quantiles = [ int(support_levels[idx] * len(df)) for idx in range(len(support_levels)) ]


# In[7]:

data_size_quantiles


# In[10]:

#%%capture

benchmark_data = [
    
]

for data_size in data_size_quantiles:
    current_rules = list_of_rules[:RULE_COUNT_TO_USE]
    current_df = df.sample(data_size)
    
    time1 = time.time()
    lambda_array = [1.0]*7     # use separate hyperparamter search routine
    s1 = smooth_local_search(current_rules, current_df, Y, lambda_array, 0.33, 0.33)
    s2 = smooth_local_search(current_rules, current_df, Y, lambda_array, 0.33, -1.0)
    f1 = func_evaluation(s1, current_rules, current_df, Y, lambda_array)
    f2 = func_evaluation(s2, current_rules, current_df, Y, lambda_array)
    
    soln_set = None
    if f1 > f2:
        soln_set = s1
    else:
        soln_set = s2
        
    time2 = time.time()
    
    duration = time2 - time1
    
    benchmark_data.append(dict(
        duration=duration,
        data_size=data_size
    ))


# In[27]:

benchmark_dataframe_lakkaraju = pd.DataFrame(benchmark_data)


# In[28]:

benchmark_dataframe_lakkaraju.to_csv("./results/titanic_data_size_benchmark_lakkaraju.csv", index=False)


# # PyIDS

# ## PyIDS setup

# In[15]:

import time


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


from sklearn.metrics import accuracy_score, auc, roc_auc_score

from pyids.ids_rule import IDSRule
from pyids.ids_ruleset import IDSRuleSet
from pyids.ids_objective_function import ObjectiveFunctionParameters, IDSObjectiveFunction
from pyids.ids_optimizer import RSOptimizer, SLSOptimizer
from pyids.ids_cacher import IDSCacher
from pyids.ids_classifier import IDS, mine_CARs


from pyarc.qcba import *

from pyarc.algorithms import createCARs, top_rules
from pyarc import TransactionDB


df = pd.read_csv("C:/code/python/interpretable_decision_sets/data/titanic.csv")
df["Died"] = df["Died"].astype(str) + "_"
cars = mine_CARs(df, rule_cutoff=100)

quant_df = QuantitativeDataFrame(df)


# ## PyIDS benchmark

# In[30]:

support_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
support_levels = list(reversed(support_levels))
data_size_quantiles = [ int(support_levels[idx] * len(df)) for idx in range(len(support_levels)) ]


# In[31]:

benchmark_data = [
    
]

for data_size in data_size_quantiles:
    current_cars = cars[:RULE_COUNT_TO_USE]
    current_df = df.sample(data_size)
    current_quant_df = QuantitativeDataFrame(current_df)
    
    times = []
    
    for _ in range(PY_IDS_DURATION_ITERATIONS):
        time1 = time.time()
        lambda_array = [1.0]*7     # use separate hyperparamter search routine
        ids = IDS()
        ids.fit(class_association_rules=current_cars, quant_dataframe=current_quant_df, debug=False)

        time2 = time.time()

        duration = time2 - time1
        times.append(duration)
        
    benchmark_data.append(dict(
        duration=np.mean(times),
        data_size=data_size
    ))


# In[32]:

benchmark_dataframe_pyids = pd.DataFrame(benchmark_data)

benchmark_dataframe_pyids


# In[33]:

#benchmark_dataframe_pyids.plot(x=["data_size"], y=["duration"])


# In[ ]:

benchmark_dataframe_pyids.to_csv("./results/titanic_data_size_benchmark_pyids.csv", index=False)

