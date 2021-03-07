from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import top_rules, createCARs

from ..data_structures.rule import IDSRule
from ..data_structures import IDSRuleSet

from ..algorithms.rule_comparator import IDSComparator
from ..model_selection import mode

import numpy as np
import random
import logging


class IDSClassifier:
    
    def __init__(self, rules):
        self.rules = rules
        self.default_class = None
        self.default_class_confidence = None
        self.quant_dataframe_train = None

        self.logger = logging.getLogger(IDSClassifier.__name__)

    def calculate_default_class(self):
        predicted_classes = self.predict(self.quant_dataframe_train)
        not_classified_idxes = [ idx for idx, val in enumerate(predicted_classes) if val == None ]
        classes = self.quant_dataframe_train.dataframe.iloc[:, -1]

        actual_classes = list(self.quant_dataframe_train.dataframe.iloc[not_classified_idxes, -1])

        # return random class
        if not list(actual_classes):
            self.default_class = random.sample(list(np.unique(classes)), 1)[0]
            self.default_class_confidence = 1

        else:
            minority_class = mode(actual_classes)

            self.default_class = minority_class
            self.default_class_confidence = actual_classes.count(minority_class) / len(actual_classes)

    def predict(self, quant_dataframe: QuantitativeDataFrame, order_type: str = "f1"):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        sorted_rules = IDSComparator().sort(self.rules, order_type=order_type)

        predicted_classes = []
    
        for _, row in quant_dataframe.dataframe.iterrows():
            appended = False
            for rule in sorted_rules:
                antecedent_dict = dict(rule.car.antecedent)
                counter = True

                for name, value in row.iteritems():
                    if name in antecedent_dict:
                        rule_value = antecedent_dict[name]

                        counter &= rule_value == value

                if counter:
                    _, predicted_class = rule.car.consequent
                    predicted_classes.append(predicted_class)

                    appended = True

                    break

            if not appended:
                predicted_classes.append(self.default_class)

        return predicted_classes

    def predict_proba(self, quant_dataframe, order_type: str = "f1"):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        confidences = []

        sorted_rules = IDSComparator().sort(self.rules, order_type=order_type)
    
        for _, row in quant_dataframe.dataframe.iterrows():
            appended = False
            for rule in sorted_rules:
                antecedent_dict = dict(rule.car.antecedent)  
                counter = True

                for name, value in row.iteritems():
                    if name in antecedent_dict:
                        rule_value = antecedent_dict[name]

                        counter &= rule_value == value

                if counter:
                    confidences.append(rule.car.confidence)

                    appended = True

                    break

            if not appended:
                confidences.append(self.default_class_confidence)

        return confidences


def mine_CARs(df, rule_cutoff, sample=False, random_seed=None, **top_rules_kwargs):
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    txns = TransactionDB.from_DataFrame(df)
    rules = top_rules(txns.string_representation, appearance=txns.appeardict, **top_rules_kwargs)
    cars = createCARs(rules)

    cars_subset = cars[:rule_cutoff]

    if sample:
        cars_subset = random.sample(cars, rule_cutoff)

    return cars_subset


def mine_IDS_ruleset(df, rule_cutoff, random_seed=None, **top_rules_kwargs):
    cars_subset = mine_CARs(df, rule_cutoff, random_seed=random_seed, **top_rules_kwargs)
    ids_rls_subset = map(IDSRule, cars_subset)
    ids_ruleset = IDSRuleSet(ids_rls_subset)

    return ids_ruleset