import numpy as np


from .ids_ruleset import IDSRuleSet
from .ids_cacher import IDSCacher

class ObjectiveFunctionParameters():
    
    def __init__(self):

        self.params = dict(
            lambda_array = 7*[1],
        )



class IDSObjectiveFunction:
    
    def __init__(self, objective_func_params=ObjectiveFunctionParameters()):
        self.objective_func_params = objective_func_params

        all_rules = self.objective_func_params.params["all_rules"]
        quant_dataframe = self.objective_func_params.params["quant_dataframe"]

        self.cacher = IDSCacher()
        self.cacher.calculate_overlap(all_rules, quant_dataframe)

    
    def f0(self, solution_set):
        all_rules = self.objective_func_params.params["all_rules"]

        f0 = len(all_rules) - len(solution_set)

        return f0
    
    def f1(self, solution_set):
        L_max = solution_set.max_rule_length()
        all_rules = self.objective_func_params.params["all_rules"]

        f1 = L_max * len(all_rules) - solution_set.sum_rule_length()

        return f1

        

    def f2(self, solution_set):
        overlap_intraclass_sum = 0

        n = len(solution_set)
        all_rules = self.objective_func_params.params["all_rules"]
        quant_dataframe = self.objective_func_params.params["quant_dataframe"]


        for i, r1 in enumerate(solution_set.ruleset):
            for j, r2 in enumerate(solution_set.ruleset):
                if i >= j:
                    continue

                if r1.car.consequent.value == r2.car.consequent.value:
                    overlap_tmp = self.cacher.overlap(r1, r2)

                    overlap_intraclass_sum += overlap_tmp
                    

        f2 = quant_dataframe.dataframe.shape[0] * len(all_rules) ** 2 - overlap_intraclass_sum

        return f2


    def f3(self, solution_set) :
        overlap_interclass_sum = 0

        n = len(solution_set)
        all_rules = self.objective_func_params.params["all_rules"]
        quant_dataframe = self.objective_func_params.params["quant_dataframe"]


        for i, r1 in enumerate(solution_set.ruleset):
            for j, r2 in enumerate(solution_set.ruleset):
                if i >= j:
                    continue

                if r1.car.consequent.value != r2.car.consequent.value:
                    overlap_tmp = self.cacher.overlap(r1, r2)

                    overlap_interclass_sum += overlap_tmp
                    

        f3 = quant_dataframe.dataframe.shape[0] * len(all_rules) ** 2 - overlap_interclass_sum

        return f3

    def f4(self, solution_set):
        classes_covered = set()

        for rule in solution_set.ruleset:
            classes_covered.add(rule.car.consequent.value)

        f4 = len(classes_covered)

        return f4


    def f5(self, solution_set):
        all_rules = self.objective_func_params.params["all_rules"]
        quant_dataframe = self.objective_func_params.params["quant_dataframe"]
        sum_incorrect_cover = 0

        for rule in solution_set.ruleset:
            sum_incorrect_cover += np.sum(rule.incorrect_cover(quant_dataframe))

        f5 = quant_dataframe.dataframe.shape[0] * len(all_rules) - sum_incorrect_cover

        return f5


    def f6(self, solution_set):
        quant_dataframe = self.objective_func_params.params["quant_dataframe"]
        correctly_covered = np.zeros(quant_dataframe.dataframe.index.size).astype(bool)

        for rule in solution_set.ruleset:
            correctly_covered = correctly_covered | rule.correct_cover(quant_dataframe)

        f6 = np.sum(correctly_covered)

        return f6


    def evaluate(self, solution_set):
        if type(solution_set) != IDSRuleSet:
            raise Exception("Type of solution_set must by IDSRuleSet")

        l = self.objective_func_params.params["lambda_array"]

        f0 = self.f0(solution_set)
        f1 = self.f1(solution_set)
        f2 = self.f2(solution_set)
        f3 = self.f3(solution_set)
        f4 = self.f4(solution_set)
        f5 = self.f5(solution_set)
        f6 = self.f6(solution_set)

        result = l[0] * f0 + l[1] * f1 + l[2] + f2 + l[3] * f3 + l[4] * f4 + l[5] * f5 + l[6] * f6


        return result



