import numpy as np
import random
import math

from pyids.data_structures.ids_ruleset import IDSRuleSet

class RSOptimizer:
    
    def __init__(self, input_set, probability=0.5, random_seed=None):

        self.input_set = input_set
        self.solution_set = set()

        if random_seed:
            np.random.seed(random_seed)
        
        self.probability = probability
        
    def optimize(self):
        self.solution_set = set()
        solution_set = set()
    
        for member in self.input_set:
            if np.random.uniform() <= self.probability:
                solution_set.add(member)
                
        
        self.solution_set = solution_set
        
        return self.solution_set






    




