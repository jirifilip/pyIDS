from pyarc.qcba.data_structures import QuantitativeDataFrame

from .ids_objective_function import IDSObjectiveFunction, ObjectiveFunctionParameters

from ..data_structures.ids_rule import IDSRule
from ..data_structures.ids_ruleset import IDSRuleSet

from .optimizers.sls_optimizer import SLSOptimizer
from .optimizers.dls_optimizer import DLSOptimizer
from .optimizers.dusm_optimizer import DeterministicUSMOptimizer
from .optimizers.rusm_optimizer import RandomizedUSMOptimizer

from ..model_selection import encode_label, calculate_ruleset_statistics, mode
from ..algorithms.ids_classifier import IDSClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

import logging
import numpy as np


class IDS:

    def __init__(self, algorithm="SLS"):
        self.clf: IDSClassifier = None
        self.cacher = None
        self.ids_ruleset = None
        self.algorithms = dict(
            SLS=SLSOptimizer,
            DLS=DLSOptimizer,
            DUSM=DeterministicUSMOptimizer,
            RUSM=RandomizedUSMOptimizer
        )

        self.algorithm = algorithm
        self.logger = logging.getLogger(IDS.__name__)

    def fit(
            self,
            quant_dataframe,
            class_association_rules=None,
            lambda_array=7*[1],
            default_class="majority_class_in_uncovered",
            optimizer_args=dict(),
            random_seed=None
    ):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        params = ObjectiveFunctionParameters()

        if not self.ids_ruleset:
            ids_rules = list(map(IDSRule, class_association_rules))
            all_rules = IDSRuleSet(ids_rules)
            params.params["all_rules"] = all_rules
        elif self.ids_ruleset and not class_association_rules:
            print("using provided ids ruleset and not class association rules")
            params.params["all_rules"] = self.ids_ruleset

        params.params["len_all_rules"] = len(params.params["all_rules"])
        params.params["quant_dataframe"] = quant_dataframe
        params.params["lambda_array"] = lambda_array

        # objective function
        objective_function = IDSObjectiveFunction(objective_func_params=params, cacher=self.cacher)

        optimizer = self.algorithms[self.algorithm](
            objective_function,
            params,
            random_seed=random_seed,
            optimizer_args=optimizer_args
        )

        solution_set = optimizer.optimize()

        self.logger.debug("Solution set optimized")

        self.clf = IDSClassifier(solution_set)
        self.clf.rules = sorted(self.clf.rules, reverse=True)
        self.clf.quant_dataframe_train = quant_dataframe

        if default_class == "majority_class_in_all":
            classes = quant_dataframe.dataframe.iloc[:, -1]
            self.clf.default_class = mode(classes)
            self.clf.default_class_confidence = classes.count(self.clf.default_class) / len(classes)
        elif default_class == "majority_class_in_uncovered":
            self.clf.calculate_default_class()

        self.logger.debug(f"Chosen default class: {self.clf.default_class}")
        self.logger.debug(f"Default class confidence: {self.clf.default_class_confidence}")

        return self

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