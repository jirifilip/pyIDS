import math
import numpy as np
import logging
from typing import Iterable

from .rs_optimizer import RSOptimizer
from ...data_structures import IDSRuleSet
from ...data_structures.rule import IDSRule


class SLSOptimizer:

    def __init__(
            self,
            objective_function,
            rules: IDSRuleSet,
    ):
        self.delta = 0.33

        self.objective_function = objective_function

        self.rules = rules

        self.rs_optimizer = RSOptimizer(rules.ruleset)

        self.logger = logging.getLogger(SLSOptimizer.__name__)
        self.max_omega_iterations = 10000

    def compute_OPT(self):
        solution_set = self.rs_optimizer.optimize()

        return self.objective_function.evaluate(IDSRuleSet(solution_set))

    def estimate_omega(self, rule, solution_set, error_threshold, delta):
        exp_include_func_vals = []
        exp_exclude_func_vals = []

        omega_estimation_extensions = 0
        omega_estimation_iterations = 10

        last_standard_error = 0
        current_standard_error = 0
        idx = 0
        improvement_rate = 1

        iteration_step_dict = dict()

        while True:

            for _ in range(omega_estimation_iterations):
                temp_soln_set = self.sample_random_set(solution_set.ruleset, delta)
                temp_soln_set.add(rule)

                func_val = self.objective_function.evaluate(IDSRuleSet(temp_soln_set))

                exp_include_func_vals.append(func_val)

            for _ in range(omega_estimation_iterations):
                temp_soln_set = self.sample_random_set(solution_set.ruleset, delta)
                if rule in temp_soln_set:
                    temp_soln_set.remove(rule)

                func_val = self.objective_function.evaluate(IDSRuleSet(temp_soln_set))

                exp_exclude_func_vals.append(func_val)

            variance_exp_include = np.var(exp_include_func_vals)
            variance_exp_exclude = np.var(exp_exclude_func_vals)
            standard_error = math.sqrt(
                variance_exp_include / len(exp_include_func_vals) + variance_exp_exclude / len(exp_exclude_func_vals))

            self.logger.debug("INFO - stardard error of omega estimate: {}".format(standard_error))

            if standard_error > error_threshold:
                if idx == 0:
                    last_standard_error = standard_error
                    idx += 1
                    continue

                current_standard_error = standard_error
                current_step = last_standard_error - current_standard_error
                remaining_step = current_standard_error - error_threshold

                improvement_rate = current_step / last_standard_error
                if improvement_rate == 0:
                    improvement_rate = 1

                iteration_step_dict[omega_estimation_iterations] = current_step

                if not remaining_step <= current_step:
                    omega_estimation_iterations = round(omega_estimation_iterations / improvement_rate) + 1

                if omega_estimation_iterations > self.max_omega_iterations:
                    omega_estimation_iterations = self.max_omega_iterations

                self.logger.debug(
                    f"INFO - current_standard_error: {current_standard_error},"
                    f" last_standard_error: {last_standard_error},"
                    f" improvement_rate: {improvement_rate}, "
                    f" omega_estimation_iterations: {omega_estimation_iterations}"
                )
                self.logger.debug("INFO - {} > {} => omega estimation continues".format(standard_error, error_threshold))

                last_standard_error = current_standard_error

            if standard_error <= error_threshold:
                self.logger.debug("INFO - omega succesfully estimated")
                break

        return np.mean(exp_include_func_vals) - np.mean(exp_exclude_func_vals)

    def optimize_delta(self, delta, delta_prime):
        OPT = self.compute_OPT()
        n = len(self.rules)

        self.logger.debug("INFO - Number of input rules: {}".format(n))
        self.logger.debug("INFO - RandomOptimizer estimated the OPTIMUM value as: {}".format(OPT))
        self.logger.debug(
            "INFO - Threshold value (2/(n*n) * OPT) = {}. This is the standard error treshold value.".format(
                2.0 / (n * n) * OPT))

        soln_set = IDSRuleSet(set())

        restart_omega_computations = False

        while True:
            omega_estimates = {}
            for rule in self.rules.ruleset:
                self.logger.debug("INFO - Estimating omega for rule: {}".format(rule))

                omega_est = self.estimate_omega(rule, soln_set, 1.0 / (n * n) * OPT, delta)
                omega_estimates[rule] = omega_est

                if rule in soln_set.ruleset:
                    continue

                if omega_est > 2.0 / (n * n) * OPT:
                    # add this element to solution set and recompute omegas
                    soln_set.ruleset.add(rule)
                    restart_omega_computations = True

                    self.logger.debug("Adding rule: {} to the solution set.".format(rule))

                    break

            if restart_omega_computations:
                restart_omega_computations = False
                continue

            for rule_idx, rule in enumerate(soln_set.ruleset):
                if omega_estimates[rule] < -2.0 / (n * n) * OPT:
                    soln_set.ruleset.remove(rule)
                    restart_omega_computations = True

                    self.logger.debug("Removing rule: {} from the solution set.".format(rule))
                    break

            if restart_omega_computations:
                restart_omega_computations = False
                continue

            return self.sample_random_set(soln_set.ruleset, delta_prime)

    def sample_random_set(self, soln_set, delta):
        # get params from cache
        return_set = set()
        all_rules_set = self.rules.ruleset

        p = (delta + 1.0) / 2
        for item in soln_set:
            random_val = np.random.uniform()
            if random_val <= p:
                return_set.add(item)

        p_prime = (1.0 - delta) / 2
        for item in (all_rules_set - soln_set):
            random_val = np.random.uniform()
            if random_val <= p_prime:
                return_set.add(item)

        return return_set

    def optimize(self):
        solution1 = self.optimize_delta(1 / 3, 1 / 3)
        solution2 = self.optimize_delta(1 / 3, -1.0)

        func_val1 = self.objective_function.evaluate(IDSRuleSet(solution1))
        func_val2 = self.objective_function.evaluate(IDSRuleSet(solution2))

        if func_val1 >= func_val2:
            return solution1
        else:
            return solution2
