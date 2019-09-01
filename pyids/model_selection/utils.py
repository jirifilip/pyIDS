import numpy as np
import pandas as pd
import random

from pyarc.algorithms import top_rules, createCARs
from pyarc.data_structures import TransactionDB
from ..data_structures.ids_rule import IDSRule
from ..data_structures.ids_ruleset import IDSRuleSet

def encode_label(actual, predicted):
    levels = set(actual)
    
    actual = np.copy(actual)
    predicted = np.copy(predicted)
    
    for idx, level in enumerate(levels):
        actual[actual == level] = idx
        predicted[predicted == level] = idx

    actual = actual.astype(int)
    predicted = predicted.astype(int)
        
    return actual, predicted


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
