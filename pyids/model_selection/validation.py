from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import top_rules, createCARs

import pandas as pd
import numpy as np



class KFoldCV:

    def __init__(self, classifier, folds_pandas_dataframes, score_auc=False):
        self.folds = folds_pandas_dataframes
        self.num_folds = len(self.folds)
        self.classifier = classifier

        self.score_auc = score_auc

        self.classifiers = []

    def _prepare_dataframes(self):

        dataframes = []

        for idx in range(self.num_folds):
            current_test_fold = self.folds[idx]
            current_train_folds = [ self.folds[i] for i in range(self.num_folds) ] 

            current_train_dataset = pd.concat(current_train_folds)

            dataframes.append((current_train_dataset, current_test_fold))

        return dataframes


    def fit(self, rule_cutoff):
        dataframes = self._prepare_dataframes()

        scores = []

        for dataframe_train, dataframe_test in dataframes:
            txns_train = TransactionDB.from_DataFrame(dataframe_train)

            rules = top_rules(txns_train.string_representation, appearance=txns_train.appeardict)
            cars = createCARs(rules)[:rule_cutoff]

            quant_dataframe_train = QuantitativeDataFrame(dataframe_train)
            quant_dataframe_test = QuantitativeDataFrame(dataframe_test)

            self.classifier.fit(quant_dataframe_train, cars, debug=self.debug)

            score = None
            
            if self.score_auc:
                score = self.classifier.score_auc(quant_dataframe_test)
            else:
                score = self.classifier.score(quant_dataframe_test)


            scores.append(score)

        return scores


