import numpy as np
from typing import Iterable, List
from abc import ABC, abstractmethod

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.data_structures import IDSRuleSet
from pyids.data_structures.cacher import IDSCacher
from pyids.data_structures.rule import IDSRule


class ObjectiveFunction(ABC):
    pass


class IDSObjectiveFunction(ObjectiveFunction):
    
    def __init__(
        self,
        dataframe: QuantitativeDataFrame,
        rules: Iterable[IDSRule],
        lambda_array: List[float] = 7 * [1],
    ):
        self.rules = rules
        self.dataframe = dataframe
        self.lambda_array = lambda_array

        self._len_rules = len(self.rules)

        all_rules_lengths = [ len(rule) for rule in rules.ruleset ]
        self._L_max = max(all_rules_lengths)

        self.cacher = IDSCacher()
        self.cacher.calculate_overlap(rules, dataframe)

    def f0(self, solution_set: IDSRuleSet):
        f0 = self._len_rules - len(solution_set)

        return f0
    
    def f1(self, solution_set: IDSRuleSet):
        f1 = self._L_max * self._len_rules - solution_set.sum_rule_length()

        return f1

    def f2(self, solution_set: IDSRuleSet):
        overlap_intraclass_sum = 0

        for i, r1 in enumerate(solution_set.ruleset):
            for j, r2 in enumerate(solution_set.ruleset):
                if i >= j:
                    continue

                if r1.car.consequent.value == r2.car.consequent.value:
                    overlap_tmp = self.cacher.overlap(r1, r2)

                    overlap_intraclass_sum += overlap_tmp
                    
        f2 = self.dataframe.dataframe.shape[0] * self._len_rules ** 2 - overlap_intraclass_sum

        return f2

    def f3(self, solution_set: IDSRuleSet) :
        overlap_interclass_sum = 0

        for i, r1 in enumerate(solution_set.ruleset):
            for j, r2 in enumerate(solution_set.ruleset):
                if i >= j:
                    continue

                if r1.car.consequent.value != r2.car.consequent.value:
                    overlap_tmp = self.cacher.overlap(r1, r2)

                    overlap_interclass_sum += overlap_tmp

        f3 = self.dataframe.dataframe.shape[0] * self._len_rules ** 2 - overlap_interclass_sum

        return f3

    def f4(self, solution_set: IDSRuleSet):
        classes_covered = set()

        for rule in solution_set.ruleset:
            classes_covered.add(rule.car.consequent.value)

        f4 = len(classes_covered)

        return f4

    def f5(self, solution_set: IDSRuleSet):
        sum_incorrect_cover = 0

        for rule in solution_set.ruleset:
            sum_incorrect_cover += np.sum(rule.incorrect_cover(self.dataframe))

        f5 = self.dataframe.dataframe.shape[0] * self._len_rules - sum_incorrect_cover

        return f5

    def f6(self, solution_set: IDSRuleSet):
        correctly_covered = np.zeros(self.dataframe.dataframe.index.size).astype(bool)

        for rule in solution_set.ruleset:
            correctly_covered = correctly_covered | rule.correct_cover(self.dataframe)

        f6 = np.sum(correctly_covered)

        return f6

    def evaluate(self, solution_set: IDSRuleSet):
        if type(solution_set) != IDSRuleSet:
            raise Exception(f"Type of solution_set must by f{IDSRuleSet.__name__}")

        l = self.lambda_array

        f0 = self.f0(solution_set) if l[0] != 0 else 0
        f1 = self.f1(solution_set) if l[1] != 0 else 0
        f2 = self.f2(solution_set) if l[2] != 0 else 0
        f3 = self.f3(solution_set) if l[3] != 0 else 0
        f4 = self.f4(solution_set) if l[4] != 0 else 0
        f5 = self.f5(solution_set) if l[5] != 0 else 0
        f6 = self.f6(solution_set) if l[6] != 0 else 0

        fs = np.array([
            f0, f1, f2, f3, f4, f5, f6
        ])

        result = np.dot(l, fs)

        return result


class NormalizedF1ObjectiveFunction(IDSObjectiveFunction):

    def __init__(self, *args, precision_weight=1, recall_weight=1, **kwargs):
        self.precision_weight = precision_weight
        self.recall_weight = recall_weight

        self.normalized_precision_func = self.get_normalized_objective(lambda x: self.f5(x))
        self.normalized_recall_func = self.get_normalized_objective(lambda x: self.f6(x))

        super(IDSObjectiveFunction, self).__init__(*args, **kwargs)

    def get_normalized_objective(self, objective_func):
        empty_ruleset = IDSRuleSet(rules=set())
        full_ruleset = self.rules

        objective_max = objective_func(empty_ruleset)
        objective_min = objective_func(full_ruleset)

        def normalized_objective_func(solution_set):
            objective_value = objective_func(solution_set)

            normalized_objective = (objective_value - objective_min) / (objective_max - objective_min)

            return normalized_objective

        return normalized_objective_func

    def evaluate(self, solution_set: IDSRuleSet):
        if type(solution_set) != IDSRuleSet:
            raise Exception(f"Type of solution_set must by f{IDSRuleSet.__name__}")

        precision_objective = self.normalized_precision_func(solution_set) * self.precision_weight
        recall_objective = self.normalized_recall_func(solution_set) * self.recall_weight

        objective = precision_objective + recall_objective

        return objective



