from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import ClassAssocationRule

from .objective_function import IDSObjectiveFunction, ObjectiveFunction

from ..data_structures.rule import IDSRule
from ..data_structures import IDSRuleSet

from .optimizers.sls_optimizer import SLSOptimizer
from .optimizers.dls_optimizer import DLSOptimizer
from .optimizers.dusm_optimizer import DeterministicUSMOptimizer
from .optimizers.rusm_optimizer import RandomizedUSMOptimizer

from ..model_selection import encode_label, calculate_ruleset_statistics, mode
from ..algorithms.classifier import IDSClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

import logging
import numpy as np
from typing import Iterable


class IDS:

    def __init__(self):
        self.clf: IDSClassifier = None
        self.dataframe: QuantitativeDataFrame = None
        self.ruleset: IDSRuleSet = None
        self.lambda_array = None
        self.objective_function = None
        self.optimizer = None
        self.default_class = None

        self.logger = logging.getLogger(IDS.__name__)

    def __get_objective_function(self, *, objective):
        if objective == "ids":
            return IDSObjectiveFunction(
                dataframe=self.dataframe,
                rules=self.ruleset,
                lambda_array=self.lambda_array
            )
        elif isinstance(objective, ObjectiveFunction):
            return objective
        else:
            raise ValueError("Could not process specified objective function.")

    def __get_optimizer(self, *, optimizer):
        if optimizer == "sls":
            return SLSOptimizer(rules=self.ruleset, objective_function=self.objective_function)
        elif optimizer == "rusm":
            return RandomizedUSMOptimizer(rules=self.ruleset, objective_function=self.objective_function)

    def fit(
            self,
            dataframe: QuantitativeDataFrame,
            rules: Iterable[ClassAssocationRule],
            lambda_array=7*[1],
            objective="ids",
            optimizer="sls",
            default_class="majority_class_in_uncovered",
    ):
        if type(dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        self.dataframe = dataframe
        self.ruleset = IDSRuleSet([ IDSRule(rule) for rule in rules ])
        self.lambda_array = lambda_array
        self.default_class = default_class

        self.objective_function = self.__get_objective_function(objective=objective)
        self.optimizer = self.__get_optimizer(optimizer=optimizer)

        solution_set = self.optimizer.optimize()

        self.logger.debug("Solution set optimized")

        self.__create_classifier(solution_set)

        return self

    def __create_classifier(self, solution_set):
        self.clf = IDSClassifier(solution_set)
        self.clf.rules = sorted(self.clf.rules, reverse=True)
        self.clf.quant_dataframe_train = self.dataframe

        if self.default_class == "majority_class_in_all":
            classes = self.dataframe.dataframe.iloc[:, -1]
            self.clf.default_class = mode(classes)
            self.clf.default_class_confidence = classes.count(self.clf.default_class) / len(classes)
        elif self.default_class == "majority_class_in_uncovered":
            self.clf.calculate_default_class()

        self.logger.debug(f"Chosen default class: {self.clf.default_class}")
        self.logger.debug(f"Default class confidence: {self.clf.default_class_confidence}")

    def predict(self, quant_dataframe: QuantitativeDataFrame, order_type="f1"):
        return self.clf.predict(quant_dataframe, order_type=order_type)

    def get_prediction_rules(self, quant_dataframe):
        return self.clf.get_prediction_rules(quant_dataframe)

    def score(self, quant_dataframe, order_type="f1", metric=accuracy_score):
        pred = self.predict(quant_dataframe, order_type=order_type)
        actual = quant_dataframe.dataframe.iloc[:, -1].values

        return metric(actual, pred)

    def _calculate_auc_for_ruleconf(self, quant_dataframe, order_type):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        confidences = self.clf.predict_proba(quant_dataframe, order_type=order_type)
        confidences_array = np.array(confidences)

        actual_classes = quant_dataframe.dataframe.iloc[:, -1].values
        predicted_classes = self.predict(quant_dataframe, order_type=order_type)

        actual, pred = encode_label(actual_classes, predicted_classes)

        corrected_confidences = np.where(np.equal(pred.astype(int), 1), confidences_array, 1 - confidences_array)

        return roc_auc_score(actual, corrected_confidences)

    def _calcutate_auc_classical(self, quant_dataframe, order_type):
        pred = self.predict(quant_dataframe, order_type=order_type)
        actual = quant_dataframe.dataframe.iloc[:, -1].values

        actual, pred = encode_label(actual, pred)

        return roc_auc_score(actual, pred)

    def score_auc(self, quant_dataframe: QuantitativeDataFrame, order_type="f1"):
        actual_classes = quant_dataframe.dataframe.iloc[:, -1].values
        predicted_classes = np.array(self.predict(quant_dataframe, order_type=order_type))
        predicted_probabilities = np.array(self.clf.predict_proba(quant_dataframe=quant_dataframe, order_type=order_type))

        distinct_classes = set(actual_classes)

        AUCs = []

        ones = np.ones_like(predicted_classes, dtype=int)
        zeroes = np.zeros_like(predicted_classes, dtype=int)

        for distinct_class in distinct_classes:
            class_predicted_probabilities = np.where(predicted_classes == distinct_class, predicted_probabilities, ones - predicted_probabilities)
            class_actual_probabilities = np.where(actual_classes == distinct_class, ones, zeroes)

            auc = roc_auc_score(class_actual_probabilities, class_predicted_probabilities, average="micro")
            AUCs.append(auc)

        auc_score = np.mean(AUCs)

        return auc_score

    def score_interpretability_metrics(self, quant_dataframe):
        stats = calculate_ruleset_statistics(self, quant_dataframe)

        return stats