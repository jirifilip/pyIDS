import numpy as np
import logging
from typing import Iterable

from pyarc.qcba.data_structures import QuantitativeDataFrame

from ...data_structures import IDSRuleSet
from ...data_structures.rule import IDSRule

from ..objective_function import IDSObjectiveFunction


class RandomizedUSMOptimizer:

    def __init__(
            self,
            objective_function: IDSObjectiveFunction,
            rules: IDSRuleSet,
    ):
        self.objective_function = objective_function
        self.rules = rules
        self.logger = logging.getLogger(RandomizedUSMOptimizer.__name__)

    def optimize(self):
        x0 = IDSRuleSet(set())
        y0 = IDSRuleSet({rule for rule in self.rules.ruleset})

        n = len(y0)

        self.logger.debug(f"Total # of rules to evaluate: {n}")

        for idx, rule in enumerate(self.rules.ruleset):
            self.logger.debug(f"Enumerating rule #{idx}: {rule}")

            a_set = IDSRuleSet(x0.ruleset | {rule})
            b_set = IDSRuleSet(y0.ruleset - {rule})

            a_value = self.objective_function.evaluate(a_set) - self.objective_function.evaluate(x0)
            b_value = self.objective_function.evaluate(b_set) - self.objective_function.evaluate(y0)

            a_max = max(a_value, 0)
            b_max = max(b_value, 0)

            self.logger.debug(f"rule #{idx}: len(X) = {len(x0)}")
            self.logger.debug(f"rule #{idx}: len(Y) = {len(y0)}")

            self.logger.debug(f"rule #{idx}: len(a_set) = {len(a_set.ruleset)}")
            self.logger.debug(f"rule #{idx}: len(b_set) = {len(b_set.ruleset)}")

            self.logger.debug(f"rule #{idx}:  a = {a_value}")
            self.logger.debug(f"rule #{idx}:  b = {b_value}")

            self.logger.debug(f"rule #{idx}: a' = {a_max}")
            self.logger.debug(f"rule #{idx}: b'= {b_max}")

            x_probability = 1

            if not (a_max == 0 and b_max == 0):
                x_probability = a_max / (a_max + b_max)
                self.logger.debug(f"x_probability for rule #{idx} = {x_probability}")

            if np.random.uniform() <= x_probability:
                self.logger.debug(f"rule #{idx} added to X")
                x0.ruleset.add(rule)
            else:
                self.logger.debug(f"rule #{idx} removed from Y")
                y0.ruleset.remove(rule)

        x_value = self.objective_function.evaluate(x0)
        y_value = self.objective_function.evaluate(y0)

        if x_value > y_value:
            self.logger.debug(f"Final ruleset length: {len(x0.ruleset)}")
            return x0.ruleset
        else:
            self.logger.debug(f"Final ruleset length: {len(y0.ruleset)}")
            return y0.ruleset
