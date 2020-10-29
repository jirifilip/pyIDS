from ...data_structures.ids_ruleset import IDSRuleSet

import numpy as np
import logging

class DLSOptimizer:

    def __init__(self, objective_function, objective_func_params, random_seed=None):
        self.objective_function_params = objective_func_params
        self.objective_function = objective_function

        self.logger = logging.Logger(DLSOptimizer.__name__)

    def find_best_element(self):
        all_rules = self.objective_function_params.params["all_rules"]
        all_rules_list = list(all_rules.ruleset)

        func_values = []

        for rule in all_rules_list:
            new_ruleset = IDSRuleSet([rule])
            func_val = self.objective_function.evaluate(new_ruleset)

            func_values.append(func_val)

        best_rule_idx = np.argmax(func_values)
        best_rule = all_rules_list[best_rule_idx]

        return best_rule

    def optimize(self):
        all_rules = self.objective_function_params.params["all_rules"]
        solution_set = self.optimize_solution_set()

        all_rules_without_solution_set = IDSRuleSet(all_rules.ruleset - solution_set.ruleset)

        func_val1 = self.objective_function.evaluate(solution_set)
        func_val2 = self.objective_function.evaluate(all_rules_without_solution_set)

        if func_val1 >= func_val2:
            self.logger.debug(f"Objective value of solution set: {func_val1}")
            return solution_set.ruleset
        else:
            self.logger.debug(f"Objective value of solution set: {func_val2}")
            return all_rules_without_solution_set.ruleset

    def optimize_solution_set(self, epsilon=0.05):
        all_rules = self.objective_function_params.params["all_rules"]
        n = len(all_rules)

        soln_set = IDSRuleSet(set())

        best_first_rule = self.find_best_element()
        soln_set.ruleset.add(best_first_rule)

        soln_set_objective_value = self.objective_function.evaluate(soln_set)

        restart_computations = False

        while True:
            for rule in all_rules.ruleset - soln_set.ruleset:
                self.logger.debug(f"Testing if rule is good to add {rule}")

                new_soln_set = IDSRuleSet(soln_set.ruleset | {rule})
                func_val = self.objective_function.evaluate(new_soln_set)

                if func_val > (1 + epsilon / (n * n)) * soln_set_objective_value:
                    soln_set.ruleset.add(rule)
                    soln_set_objective_value = func_val
                    restart_computations = True

                    self.logger.debug(f"Adding to the solution set rule {rule}")
                    break

            if restart_computations:
                restart_computations = False
                continue

            for rule in soln_set.ruleset:
                self.logger.debug(f"Testing should remove rule {rule}")

                new_soln_set = IDSRuleSet(soln_set.ruleset - {rule})
                func_val = self.objective_function.evaluate(new_soln_set)

                if func_val > (1 + epsilon / (n * n)) * soln_set_objective_value:
                    # add this element to solution set and recompute omegas
                    soln_set.ruleset.add(rule)
                    soln_set_objective_value = func_val
                    restart_computations = True

                    self.logger.debug(f"Removing from solution set rule {rule}")
                    break

            if restart_computations:
                restart_computations = False
                continue

            return soln_set