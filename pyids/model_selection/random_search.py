import numpy as np
import itertools

from .param_space_optimizer import ParameterSpaceOptimizer

from typing import List, Dict, Tuple
import random


class RandomSearch(ParameterSpaceOptimizer):

    def __init__(
            self,
            func,
            func_args_ranges: Dict[str, Tuple[int, int]],
            max_iterations=500,
    ):

        self.func = func
        self.func_args_ranges = func_args_ranges
        self.max_iterations = max_iterations

        self.procedure_data = []

    def fit(self):
        current_best_params = self.sample_starting_params()
        current_best_func_value = self.func(current_best_params)

        for i in range(self.max_iterations):
            new_params = current_best_params.copy()

            for param_name in self.func_args_ranges.keys():
                lower_bound, upper_bound = self.func_args_ranges[param_name]

                max_increase = upper_bound - lower_bound

                new_param = lower_bound + random.random() * max_increase

                new_params[param_name] = new_param

            new_func_value = self.func(new_params)

            if new_func_value > current_best_func_value:
                current_best_func_value = new_func_value
                current_best_params = new_params.copy()

        return current_best_params



