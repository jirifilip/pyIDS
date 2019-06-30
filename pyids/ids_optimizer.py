import numpy as np
import random
import math

from .ids_ruleset import IDSRuleSet

class RSOptimizer:
    
    def __init__(self, input_set, probability=0.5):
        self.input_set = input_set
        self.solution_set = set()
        
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
    
    def __init__(self, objective_function, objective_func_params):
        self.delta = 0.33
        self.objective_function_params = objective_func_params 
        self.objective_function = objective_function
        self.rs_optimizer = RSOptimizer(self.objective_function_params.params["all_rules"].ruleset)
    
    def compute_OPT(self):
        solution_set = self.rs_optimizer.optimize()
        return self.objective_function.evaluate(IDSRuleSet(solution_set))
    
    def estimate_omega_for_element(self, rule, solution_set, error_threshold, delta):
        all_rules = self.objective_function_params.params["all_rules"]

        Exp1_func_vals = []

        Exp2_func_vals = []

        while(True):

            # first expectation term (include x)
            for i in range(10):
                temp_soln_set = self.sample_random_set(solution_set.ruleset, delta)
                temp_soln_set.add(rule)
                
                func_val = self.objective_function.evaluate(IDSRuleSet(temp_soln_set))

                Exp1_func_vals.append(func_val)

            # second expectation term (exclude x)
            for j in range(10):
                temp_soln_set = self.sample_random_set(solution_set.ruleset, delta)
                if rule in temp_soln_set:
                    temp_soln_set.remove(rule)

                func_val = self.objective_function.evaluate(IDSRuleSet(temp_soln_set))

                Exp2_func_vals.append(func_val)

            # compute standard error of mean difference
            variance_Exp1 = np.var(Exp1_func_vals)
            variance_Exp2 = np.var(Exp2_func_vals)
            std_err = math.sqrt(variance_Exp1/len(Exp1_func_vals) + variance_Exp2/len(Exp2_func_vals))
            print("Standard Error "+str(std_err))

            if std_err <= error_threshold:
                break

        return np.mean(Exp1_func_vals) - np.mean(Exp2_func_vals)
    
    def sample_random_set(self, soln_set, delta):
        # get params from cache
        return_set = set()
        all_rules_set = self.objective_function_params.params["all_rules"].ruleset

        # sample in-set elements with prob. (delta + 1)/2
        p = (delta + 1.0)/2
        for item in soln_set:
            random_val = np.random.uniform()
            if random_val <= p:
                return_set.add(item)

        # sample out-set elements with prob. (1 - delta)/2
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


    
    def optimize_delta(self, delta, delta_prime):
        all_rules = self.objective_function_params.params["all_rules"]
        print("all_rules")
        OPT = self.compute_OPT()
        print("OPT")
        n = len(all_rules)

        soln_set = IDSRuleSet(set())
        
        print("2/(n*n) OPT value is ", 2.0/(n*n)*OPT)

        
        restart_omega_computations = False
    
        while(True):
            print("1")

            # step 2 & 3: for each element estimate omega within certain error_threshold; if estimated omega > 2/n^2 * OPT, then add 
            # the corresponding rule to soln set and recompute omega estimates again
            omega_estimates = []
            for rule in all_rules.ruleset:

                print("Estimating omega for rule "+str(rule))
                omega_est = self.estimate_omega_for_element(rule, soln_set, 1.0/(n*n) * OPT, delta)
                omega_estimates.append(omega_est)
                #print("Omega estimate is "+str(omega_est))

                if rule in soln_set.ruleset:
                    continue

                if omega_est > 2.0/(n*n) * OPT:
                    # add this element to solution set and recompute omegas
                    soln_set.ruleset.add(rule)
                    restart_omega_computations = True
                    print("-----------------------")
                    print("Adding to the solution set rule "+str(rule))
                    print("-----------------------")
                    break    

            if restart_omega_computations: 
                restart_omega_computations = False
                continue

            # reaching this point of code means there is nothing more to add to the solution set, but we can remove elements
            for rule_ind, rule in enumerate(soln_set.ruleset):
                if omega_estimates[rule_ind] < -2.0/(n*n) * OPT:
                    soln_set.ruleset.remove(rule_ind)
                    restart_omega_computations = True

                    print("Removing from the solution set rule "+str(rule_ind))
                    break

            if restart_omega_computations: 
                restart_omega_computations = False
                continue

            # reaching here means there is no element to add or remove from the solution set
            return self.sample_random_set(soln_set.ruleset, delta_prime)
   
    
