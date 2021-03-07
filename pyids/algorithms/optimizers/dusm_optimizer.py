from ...data_structures import IDSRuleSet
from ..objective_function import IDSObjectiveFunction, ObjectiveFunctionParameters

import numpy as np
import logging


class DeterministicUSMOptimizer:

    def __init__(
            self,
            objective_function: IDSObjectiveFunction,
            objective_func_params: ObjectiveFunctionParameters,
            optimizer_args=dict(),
            random_seed=None
        ):

        self.objective_function_params = objective_func_params
        self.objective_function = objective_function

        self.logger = logging.getLogger(DeterministicUSMOptimizer.__name__)

    def optimize(self):
        all_rules = self.objective_function_params.params["all_rules"]

        x0 = set()
        y0 = set(all_rules.ruleset)

        n = len(y0)

        self.logger.debug(f"Total # of rules to evaluate: {n}")

        for idx, rule in enumerate(all_rules.ruleset):
            self.logger.debug(f"Enumerating rule #{idx}: {rule}")

            a_set = IDSRuleSet(x0 | {rule})
            b_set = IDSRuleSet(y0 - {rule})

            a_value = self.objective_function.evaluate(a_set) - self.objective_function.evaluate(IDSRuleSet(x0))
            b_value = self.objective_function.evaluate(b_set) - self.objective_function.evaluate(IDSRuleSet(y0))

            self.logger.debug(f"rule #{idx}: len(X) = {len(x0)}")
            self.logger.debug(f"rule #{idx}: len(Y) = {len(y0)}")

            self.logger.debug(f"rule #{idx}: len(a_set) = {len(a_set.ruleset)}")
            self.logger.debug(f"rule #{idx}: len(b_set) = {len(b_set.ruleset)}")

            self.logger.debug(f"rule #{idx}:  a = {a_value}")
            self.logger.debug(f"rule #{idx}:  b = {b_value}")

            if a_value > b_value:
                self.logger.debug(f"rule #{idx} added to X")
                x0.add(rule)
            else:
                self.logger.debug(f"rule #{idx} removed from Y")
                y0.remove(rule)

        x_value = self.objective_function.evaluate(IDSRuleSet(x0))
        y_value = self.objective_function.evaluate(IDSRuleSet(y0))

        if x_value > y_value:
            self.logger.debug(f"Final ruleset length: {len(x0)}")
            return x0
        else:
            self.logger.debug(f"Final ruleset length: {len(y0)}")
            return y0
