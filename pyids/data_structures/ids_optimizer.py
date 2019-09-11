import numpy as np
import random
import math

from .ids_ruleset import IDSRuleSet

class RSOptimizer:
    
    def __init__(self, input_set, probability=0.5, random_seed=None):
        print("RANDOM SEED", random_seed)

        self.input_set = input_set
        self.solution_set = set()

        if random_seed:
            random.seed(random_seed)
        
        self.probability = probability
        
    def optimize(self):
        self.solution_set = set()
        solution_set = set()
    
        for member in self.input_set:
            if random.random() <= self.probability:
                solution_set.add(member)
                
        
        self.solution_set = solution_set
        
        return self.solution_set




class SLSOptimizer:
    
    def __init__(self, objective_function, objective_func_params, debug=True, random_seed=None):
        print("RANDOM SEED", random_seed)
        
        self.delta = 0.33
        self.objective_function_params = objective_func_params 
        self.objective_function = objective_function
        self.rs_optimizer = RSOptimizer(self.objective_function_params.params["all_rules"].ruleset, random_seed=random_seed)
        self.debug = debug

        if random_seed:
            np.random.seed(random_seed)
    
    def compute_OPT(self):
        solution_set = self.rs_optimizer.optimize()
        
        return self.objective_function.evaluate(IDSRuleSet(solution_set))


    def estimate_omega(self, rule, solution_set, error_threshold, delta):
        exp_include_func_vals = []
        exp_exclude_func_vals = []

        while True:

            for _ in range(10):
                temp_soln_set = self.sample_random_set(solution_set.ruleset, delta)
                temp_soln_set.add(rule)
                
                func_val = self.objective_function.evaluate(IDSRuleSet(temp_soln_set))

                exp_include_func_vals.append(func_val)

            for _ in range(10):
                temp_soln_set = self.sample_random_set(solution_set.ruleset, delta)
                if rule in temp_soln_set:
                    temp_soln_set.remove(rule)

                func_val = self.objective_function.evaluate(IDSRuleSet(temp_soln_set))

                exp_exclude_func_vals.append(func_val)

            variance_exp_include = np.var(exp_include_func_vals)
            variance_exp_exclude = np.var(exp_exclude_func_vals)
            standard_error = math.sqrt(variance_exp_include/len(exp_include_func_vals) + variance_exp_exclude/len(exp_exclude_func_vals))
            
            if self.debug:
                print("Standard Error", standard_error)

            if standard_error <= error_threshold:
                break

        return np.mean(exp_include_func_vals) - np.mean(exp_exclude_func_vals)
    
    def optimize_delta(self, delta, delta_prime):
        all_rules = self.objective_function_params.params["all_rules"]
        OPT = self.compute_OPT()
        n = len(all_rules)

        soln_set = IDSRuleSet(set())

        if self.debug:        
            print("2/(n*n) * OPTIMUM VALUE =", 2.0/(n*n)*OPT)

        
        restart_omega_computations = False
    
        while True:
            omega_estimates = []
            for rule in all_rules.ruleset:

                if self.debug:
                    print("Estimating omega for rule", rule, sep="\n")
                
                omega_est = self.estimate_omega(rule, soln_set, 0.5*1/(n*n) * OPT, delta)
                #omega_est = self.estimate_omega(rule, soln_set, 1/(n*n) * OPT, delta)
                omega_estimates.append(omega_est)

                if rule in soln_set.ruleset:
                    continue

                if omega_est > 2.0/(n*n) * OPT:
                    # add this element to solution set and recompute omegas
                    soln_set.ruleset.add(rule)
                    restart_omega_computations = True
                    
                    if self.debug:
                        print("adding rule to solution set")
                    break    

            if restart_omega_computations: 
                restart_omega_computations = False
                continue

            for rule_idx, rule in enumerate(soln_set.ruleset):
                if omega_estimates[rule_idx] < -2.0/(n*n) * OPT:
                    soln_set.ruleset.remove(rule_idx)
                    restart_omega_computations = True

                    if self.debug:
                        print("removing rule from solution set")
                    break

            if restart_omega_computations: 
                restart_omega_computations = False
                continue

            return self.sample_random_set(soln_set.ruleset, delta_prime)


    def sample_random_set(self, soln_set, delta):
        # get params from cache
        return_set = set()
        all_rules_set = self.objective_function_params.params["all_rules"].ruleset

        p = (delta + 1.0)/2
        for item in soln_set:
            random_val = np.random.uniform()
            if random_val <= p:
                return_set.add(item)

        p_prime = (1.0 - delta)/2
        for item in (all_rules_set - soln_set):
            random_val = np.random.uniform()
            if random_val <= p_prime:
                return_set.add(item)

        return return_set
    
    
    def optimize(self):
        solution1 = self.optimize_delta(1/3, 1/3)
        solution2 = self.optimize_delta(1/3, -1.0)

        func_val1 = self.objective_function.evaluate(IDSRuleSet(solution1))
        func_val2 = self.objective_function.evaluate(IDSRuleSet(solution2))

        if func_val1 >= func_val2:
            return solution1
        else:
            return solution2


    

# Deterministic Local Search
class DLSOptimizer:
    
    def __init__(self, objective_function, objective_func_params, debug=True, random_seed=None):
        self.objective_function_params = objective_func_params 
        self.objective_function = objective_function
        self.debug = debug
    
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
            if self.debug:
                print("Objective value of solution set:", func_val1)
            return solution_set.ruleset
        else:
            if self.debug:
                print("Objective value of solution set:", func_val2)
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
                if self.debug:
                    print("Testing if rule is good to add "+ str(rule))
                
                new_soln_set = IDSRuleSet(soln_set.ruleset | {rule})
                func_val = self.objective_function.evaluate(new_soln_set)

                if func_val > (1 + epsilon/(n*n)) * soln_set_objective_value:
                    # add this element to solution set and recompute omegas
                    soln_set.ruleset.add(rule)
                    soln_set_objective_value = func_val
                    restart_computations = True
                    
                    if self.debug:
                        print("-----------------------")
                        print("Adding to the solution set rule "+ str(rule))
                        print("-----------------------")
                    break    

            if restart_computations: 
                restart_computations = False
                continue



            for rule in soln_set.ruleset:
                if self.debug:
                    print("Testing should remove rule "+str(rule))
                
                new_soln_set = IDSRuleSet(soln_set.ruleset - {rule})
                func_val = self.objective_function.evaluate(new_soln_set)

                if func_val > (1 + epsilon/(n*n)) * soln_set_objective_value:
                    # add this element to solution set and recompute omegas
                    soln_set.ruleset.add(rule)
                    soln_set_objective_value = func_val
                    restart_computations = True
                    
                    if self.debug:
                        print("-----------------------")
                        print("Removing from solution set rule "+str(rule))
                        print("-----------------------")
                    break    

            if restart_computations: 
                restart_computations = False
                continue

            return soln_set
    
