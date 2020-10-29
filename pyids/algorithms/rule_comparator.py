from ..data_structures.ids_rule import IDSRule

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class IDSComparator:

    def __init__(self):
        self.comparators = dict(
            f1=RuleComparatorF1,
            cba=RuleComparatorCBA,
            random=RuleComparatorRandom
        )

    def sort(self, rules: List[IDSRule], order_type: str = "f1"):

        comparator = self.comparators[order_type](rules)
        sorted_rules = comparator.sort()

        return sorted_rules


class RuleComparator(ABC):

    def __init__(self, rules: List[IDSRule]):
        self.rules = rules

    def sort(self):
        sorted_rules = sorted(
            self.rules,
            key=lambda rule: self.order(rule),
            reverse=True
        )

        return sorted_rules

    @abstractmethod
    def order(self, rule: IDSRule):
        pass


class RuleComparatorF1(RuleComparator):

    def order(self, rule: IDSRule):
        f1_score = rule.f1

        return f1_score


class RuleComparatorCBA(RuleComparator):

    def order(self, rule: IDSRule):
        """
        precedence operator. Determines if this rule
        has higher precedence. Rules are sorted according
        to their confidence, support and length.
        """

        return (
            rule.car.confidence,
            rule.car.support,
            rule.car.rulelen
        )


class RuleComparatorRandom(RuleComparator):

    def order(self, rule: IDSRule):
        """
        precedence operator. Determines if this rule
        has higher precedence. Rules are sorted randomly.
        """

        return np.random.random() <= 0.5



