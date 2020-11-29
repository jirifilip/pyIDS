from .param_space_optimizer import ParameterSpaceOptimizer
from typing import Tuple, Dict, List
import pandas as pd

def _ternary_search(func, left, right, absolute_precision, debug=False):
    """
    taken from wikipedia article on ternary search
    """
    
    while True:
        if abs(right - left) < absolute_precision:
            return (left + right)/2

        left_third = left + (right - left)/3
        right_third = right - (right - left)/3
        
        if debug:
            print(left_third, right_third)

        if func(left_third) < func(right_third):
            left = left_third
        else:
            right = right_third


class CoordinateAscent(ParameterSpaceOptimizer):

    def __init__(
            self,
            func,
            func_args_ranges: Dict[str, Tuple[int, int]],
            func_args_extension: Dict[str, int] = None,
            extension_precision=50,
            ternary_search_precision=10,
            max_iterations=500,
    ):
        self.func = func
        self.func_args_ranges = func_args_ranges

        arg_names = list(self.func_args_ranges.keys())

        if func_args_extension:
            self.func_args_extension = func_args_extension
        else:
            extensions_values = len(arg_names) * [500]

            self.func_args_extension = dict(zip(arg_names, extensions_values))

        self.ternary_search_precision = dict(zip(arg_names, len(arg_names) * [ternary_search_precision]))

        self.extension_precision = extension_precision

        self.max_iterations = max_iterations

        self.procedure_data = []

    def make_1arg_func(self, variable_arg_name, fixed_params):
        def func(x):
            fixed_params_copy = fixed_params.copy()

            fixed_params_copy[variable_arg_name] = x

            return self.func(fixed_params_copy)

        return func

    def extend_interval(self, arg_name, current_value):
        lower_interval_value, upper_interval_value = self.func_args_ranges[arg_name]

        if abs(upper_interval_value - current_value) <= self.extension_precision:
            new_upper_interval_value = upper_interval_value + self.func_args_extension[arg_name]

            self.func_args_ranges[arg_name] = lower_interval_value, new_upper_interval_value

    def fit(self):
        current_params = self.sample_starting_params()

        current_procedure_data = dict()
        current_procedure_data.update(dict(
            iteration=-1,
            current_lambda_param="None",
            loss=self.func(current_params),
            current_params=current_params.copy()
        ))

        self.procedure_data.append(current_procedure_data)

        for i in range(self.max_iterations):
            for arg_name in self.func_args_ranges.keys():
                arg_func = self.make_1arg_func(arg_name, current_params)

                print(f"using precision {self.ternary_search_precision[arg_name]}")

                interval_lower, interval_upper = self.func_args_ranges[arg_name]
                best_param = _ternary_search(
                    arg_func,
                    interval_lower,
                    interval_upper,
                    self.ternary_search_precision[arg_name]
                )

                self.extend_interval(arg_name, best_param)

                _, interval_upper_new = self.func_args_ranges[arg_name]

                if interval_upper == interval_upper_new:
                    self.ternary_search_precision[arg_name] /= 2

                current_params[arg_name] = best_param

                current_procedure_data = dict()
                current_procedure_data.update(dict(
                    iteration=i,
                    current_lambda_param=arg_name,
                    loss=self.func(current_params),
                    current_params=current_params.copy()
                ))

                self.procedure_data.append(current_procedure_data)

        procedure_data_df = pd.DataFrame(self.procedure_data)
        best_loss_mask = procedure_data_df["loss"] == procedure_data_df["loss"].max()
        best_lambda_index = procedure_data_df[best_loss_mask].index[0]

        best_lambda = list(self.procedure_data[best_lambda_index]["current_params"].values())

        return best_lambda






    

