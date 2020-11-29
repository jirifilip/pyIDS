import numpy as np
import pandas as pd
import random

from pyarc.algorithms import top_rules, createCARs
from pyarc.data_structures import TransactionDB
from ..data_structures.ids_rule import IDSRule
from ..data_structures.ids_ruleset import IDSRuleSet


def encode_label(actual, predicted):
    levels = set(actual) | set(predicted)

    actual_copy = np.copy(actual)
    predicted_copy = np.copy(predicted)

    for idx, level in enumerate(sorted(levels)):
        actual_copy[actual == level] = idx
        predicted_copy[predicted == level] = idx

    actual_copy = actual_copy.astype(int)
    predicted_copy = predicted_copy.astype(int)

    return actual_copy, predicted_copy


def mode(array):
    values, counts = np.unique(array, return_counts=True)
    idx = np.argmax(counts)
    
    return values[idx] 

def train_test_split_pd(dataframe, prop=0.25):
    n = len(dataframe)
    samp = list(range(n))
    test_n = int(prop * n)
    train_n = n - test_n
    
    test_ind = random.sample(samp, test_n)
    train_ind = list(set(samp).difference(set(test_ind)))

    return dataframe.iloc[train_ind, :].reset_index(drop=True), dataframe.iloc[test_ind, :].reset_index(drop=True)
