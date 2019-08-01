from ..ids_ruleset import IDSRuleSet
from .metrics import calculate_ruleset_statistics


class ParameterSpaceOptimizer:

    def _debug(self, value, description=""):
        if self.debug:
            if description:
                print(description, value)
            else:
                print(value)

    def check_if_satisfies_interpretablity_conditions(self, rules, quant_dataframe):
        ruleset = IDSRuleSet(rules)

        metrics = calculate_ruleset_statistics(ruleset, quant_dataframe)

        condition = True
        condition = condition and metrics["fraction_overlap"] <= self.interpretability_conditions["fraction_overlap"]
        condition = condition and metrics["fraction_uncovered"] <= self.interpretability_conditions["fraction_uncovered"]
        condition = condition and metrics["average_rule_width"] <= self.interpretability_conditions["average_rule_width"]
        condition = condition and metrics["ruleset_length"] <= self.interpretability_conditions["ruleset_length"]
        condition = condition and metrics["fraction_classes"] >= self.interpretability_conditions["fraction_classes"]

        return condition

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