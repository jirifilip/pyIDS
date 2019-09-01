import numpy as np

from ..data_structures.ids_cacher import IDSCacher







def _ternary_search(func, left, right, absolute_precision, debug=False):
    """
    taken from wikipedia article from ternary search
    """
    
    while True:
        #left and right are the current bounds; the maximum is between them
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



class CoordinateAscentOptimizer:
    
    def __init__(self,
                 classifier,
                 params_len=7,
                 param_lower_bound=1, 
                 param_upper_bound=1000, 
                 maximum_delta_between_iterations=1000, 
                 maximum_consecutive_iterations=10, 
                 maximum_upper_bound_extension_iterations=5, 
                 maximum_score_estimation_iterations=1, 
                 score_estimation_function=max,
                 upper_bound_extension_precision=50,
                 ternary_search_precision=100, 
                 debug=False):

        self.ranges = 1, 1000

        self.classifier = classifier
        self.classifier_cache = IDSCacher()

        self.debug = debug
        self.params_len = params_len


        # maximum delta of parameters between iterations
        self.maximum_delta_between_iterations = maximum_delta_between_iterations
        self.maximum_consecutive_interations = maximum_consecutive_iterations
        self.maximum_upper_bound_extension_iterations = maximum_upper_bound_extension_iterations
        self.maximum_score_estimation_iterations = maximum_score_estimation_iterations
        self.score_estimation_function = score_estimation_function        

        self.upper_bound_extension_precision = upper_bound_extension_precision
        self.UPPER_BOUND_EXTENSION_NUM = 500

        self.ternary_search_precision = ternary_search_precision

        self.current_best_params = np.array(params_len * [param_lower_bound])

        self.classifier_params_ranges = []
        for _ in range(params_len):
            self.classifier_params_ranges.append(self.ranges)
        self.classifier_params_ranges = np.array(self.classifier_params_ranges)

        self._debug(self.classifier_params_ranges)

    def _debug(self, value, description=""):
        if self.debug:
            if description:
                print(description, value)
            else:
                print(value)
        

    def _prepare(self, ids_ruleset, quant_dataframe):
        self.classifier_cache.calculate_overlap(ids_ruleset, quant_dataframe)

        self.classifier.cache = self.classifier_cache
        self.classifier.ids_ruleset = ids_ruleset

    def estimate_classifier_score(self, lambda_array, quant_dataframe_train, quant_dataframe_test):
        self._debug("estimating score")
        
        estimates = []

        for i in range(self.maximum_score_estimation_iterations):
            self._debug(i, "score estimation iteration:")
            
            self.classifier.fit(quant_dataframe_train, debug=False, lambda_array=lambda_array)
            score = self.classifier.score_auc(quant_dataframe_test)

            self._debug(score, "iteration {} score:".format(i))

            estimates.append(score)


        final_score = self.score_estimation_function(estimates)
        self._debug(final_score, "score:")

        return final_score

    def extend_search_interval(self, current_best_param_index, upper_bound, function_to_optimize):
        consecutive_iterations = 0

        while abs(self.current_best_params[current_best_param_index] - upper_bound) <= self.upper_bound_extension_precision and consecutive_iterations <= self.maximum_upper_bound_extension_iterations:
            consecutive_iterations += 1

            upper_bound += self.UPPER_BOUND_EXTENSION_NUM
            previous_best_param = self.current_best_params[current_best_param_index]

            _ternary_search(function_to_optimize, previous_best_param, upper_bound, self.ternary_search_precision)




    def fit(self, ids_ruleset, quant_dataframe_train, quant_dataframe_test):
        #
        # add type checking
        #

        self._prepare(ids_ruleset, quant_dataframe_train)

        # to compare delta between iteration
        current_best_params_previous_iteration = self.current_best_params

        # to ensure the loop will not stop at first cycle
        current_delta_between_iterations = np.array([self.maximum_delta_between_iterations] * self.params_len) + self.current_best_params  
        consecutive_iterations = 0

        while (np.abs(current_delta_between_iterations - self.current_best_params) >= self.maximum_delta_between_iterations).any() and consecutive_iterations <= self.maximum_consecutive_interations:
            consecutive_iterations += 1
            self._debug(consecutive_iterations, "consecutive iterations")
            
            for idx, param_range in enumerate(self.classifier_params_ranges):

                def function_to_optimize(param, clf=self.classifier, quant_dataframe_train=quant_dataframe_train, quant_dataframe_test=quant_dataframe_test, current_best_params=self.current_best_params):
                    current_best_params[idx] = param

                    self._debug(idx, "idx")
                    self._debug(current_best_params)
                    
                    score = self.estimate_classifier_score(current_best_params, quant_dataframe_train, quant_dataframe_test)

                    return score

                
                _ternary_search(function_to_optimize, param_range[0], param_range[1], self.ternary_search_precision)

                self.extend_search_interval(idx, param_range[1], function_to_optimize)



            current_delta_between_iterations = np.abs(current_best_params_previous_iteration - self.current_best_params)
            current_best_params_previous_iteration = self.current_best_params



        return self.current_best_params
        



    

