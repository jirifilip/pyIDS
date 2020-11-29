from ...data_structures.ids_ruleset import IDSRuleSet

import numpy as np
import logging


class RandomizedUSMOptimizer:

    def __init__(self, objective_function, objective_func_params, optimizer_args = dict(), random_seed=None):
        self.objective_function_params = objective_func_params
        self.objective_function = objective_function
        self.logger = logging.getLogger(RandomizedUSMOptimizer.__name__)

        if random_seed:
            np.random.seed(random_seed)

    def optimize(self):
        all_rules = self.objective_function_params.params["all_rules"]

        x0 = IDSRuleSet(set())
        y0 = IDSRuleSet({rule for rule in all_rules.ruleset})

        n = len(y0)

        self.logger.debug(f"Total # of rules to evaluate: {n}")

        for idx, rule in enumerate(all_rules.ruleset):
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
