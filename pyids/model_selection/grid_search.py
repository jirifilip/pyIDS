import numpy as np
import itertools
from typing import Tuple, Dict

from .param_space_optimizer import ParameterSpaceOptimizer


class GridSearch(ParameterSpaceOptimizer):
    
    def __init__(
            self,
            func,
            func_args_spaces: Dict[str, Tuple[int, int]],
            max_iterations=500
        ):

        self.func = func
        self.func_args_spaces = func_args_spaces
        self.max_iterations = max_iterations

        self.procedure_data = []

        param_spaces = []

        for arg_name, arg_space in self.func_args_spaces.items():
            param_spaces.append(arg_space)

        self.params_array_generator = itertools.product(*param_spaces)

    def fit(self):
        self.score_params_dict = dict()
        parameter_names = list(self.func_args_spaces.keys())
        
        current_iteration = 0
        
        for lambda_params in self.params_array_generator:
            current_lambda_params = dict(zip(parameter_names, lambda_params))
            
            score = self.func(current_lambda_params)

            self.score_params_dict.update({score: dict(params=lambda_params, score=score)})
            
            if current_iteration >= self.max_iterations:
                break
                
            current_iteration += 1
                
                
        maximum_score = max(self.score_params_dict.keys())
        self.best_params = self.score_params_dict[maximum_score]
        
        return self.best_params