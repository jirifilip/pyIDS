from pyarc.data_structures import TransactionDB
from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.algorithms import createCARs

import fim
import numpy as np

class RuleMiner:

    def mine_frequent_itemsets(self, pandas_df, minsup):
        txns_classless = TransactionDB.from_DataFrame(pandas_df.iloc[:,:-1])

        frequent_itemsets = fim.apriori(txns_classless.string_representation, supp=minsup*100, report="s")
        
        return frequent_itemsets

    def _convert_to_fim_rules(self, fim_itemsets, class_values, class_label="class"):
        fim_rules = []

        for itemset in fim_itemsets:
            antecedent, support = itemset

            for class_value in class_values:
                consequent = "{}:=:{}".format(class_label, class_value)
                confidence = 0

                fim_rule = consequent, antecedent, support, confidence
                fim_rules.append(fim_rule)

        return fim_rules

    def _calculate_rule_confidence(self, car, pandas_df):

        quant_dataframe = QuantitativeDataFrame(pandas_df)

        support, confidence = quant_dataframe.calculate_rule_statistics(car)

        return confidence




    def mine_rules(self, pandas_df, minsup=0.2):
        frequent_itemsets = self.mine_frequent_itemsets(pandas_df, minsup)

        distinct_classes = list(pandas_df.iloc[:,-1].unique())

        fim_rules = self._convert_to_fim_rules(frequent_itemsets, distinct_classes)

        cars = createCARs(fim_rules)

        for car in cars:
            car.confidence = self._calculate_rule_confidence(car, pandas_df)

        return cars
        



