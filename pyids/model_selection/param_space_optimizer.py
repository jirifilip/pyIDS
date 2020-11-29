import random

class ParameterSpaceOptimizer:

    def sample_starting_params(self):
        starting_params = dict()

        for arg_name in self.func_args_ranges.keys():
            interval_low, interval_up = self.func_args_ranges[arg_name]

            random_param = random.randint(interval_low, interval_up)

            starting_params[arg_name] = random_param

        return starting_params