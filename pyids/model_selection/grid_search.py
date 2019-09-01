import numpy as np
import itertools

from ..data_structures.ids_cacher import IDSCacher
from ..data_structures.ids_ruleset import IDSRuleSet

from .metrics import calculate_ruleset_statistics
from .param_space_optimizer import ParameterSpaceOptimizer



class GridSearchOptimizer(ParameterSpaceOptimizer):
    
    def __init__(self,
                 classifier,
                 params_len=7,
                 param_lower_bound=1, 
                 param_upper_bound=1000, 
                 param_search_precision=20,
                 maximum_delta_between_iterations=1000, 
                 maximum_consecutive_iterations=10, 
                 maximum_score_estimation_iterations=1,
                 maximum_iterations=1000,
                 score_estimation_function=max,
                 interpretability_conditions=dict(
                     fraction_overlap=0.10,
                     fraction_uncovered=0.15,
                     average_rule_width=8,
                     ruleset_length=10,
                     fraction_classes=1.0
                 ),
                 debug=False):

        self.debug = debug

        self.interpretability_conditions = interpretability_conditions

        self.classifier = classifier
        self.classifier_cache = IDSCacher()
        
        self.params_len = params_len
        self.precision = param_search_precision
        self.params_range = param_lower_bound, param_upper_bound
        
        self.maximum_iterations = maximum_iterations
        self.maximum_score_estimation_iterations = maximum_score_estimation_iterations
        self.score_estimation_function = score_estimation_function  

        param_spaces = []

        for i in range(self.params_len):
            param_space = np.linspace(self.params_range[0], self.params_range[1], 1000//self.precision)
            param_spaces.append(param_space)

        self.params_array_generator = itertools.product(*param_spaces)



    def _prepare(self, ids_ruleset, quant_dataframe):
        self.classifier_cache.calculate_overlap(ids_ruleset, quant_dataframe)

        self.classifier.cache = self.classifier_cache
        self.classifier.ids_ruleset = ids_ruleset




    def fit(self, ids_ruleset, quant_dataframe_train, quant_dataframe_test):
        #
        # add type checking
        #

        self._prepare(ids_ruleset, quant_dataframe_train)
        self.score_params_dict = dict() 
        
        
        current_iteration = 0
        
        for lambda_params in self.params_array_generator:
            print("curent lambda array:", lambda_params)
            
            self.classifier.fit(quant_dataframe_train, lambda_array=lambda_params, debug=self.debug)
            score = self.classifier.score_auc(quant_dataframe_test)
            satisfies_interpretability_conditions = self.check_if_satisfies_interpretablity_conditions(self.classifier.clf.rules, quant_dataframe_test)

            self.score_params_dict.update({score: dict(params=lambda_params, satisfies_interpretability_conditions=satisfies_interpretability_conditions, score=score)})
            
            if current_iteration >= self.maximum_iterations:
                break
                
            current_iteration += 1
                
                
        maximum_score = max(self.score_params_dict.keys())
        self.best_params = self.score_params_dict[maximum_score]
        
        return self.best_params