import numpy as np
import random
import math

from pyids.data_structures import IDSRuleSet


class RSOptimizer:
    
    def __init__(self, input_set, probability=0.5):

        self.input_set = input_set
        self.solution_set = set()

        self.probability = probability
        
    def optimize(self):
        self.solution_set = set()
        solution_set = set()
    
        for member in self.input_set:
            if np.random.uniform() <= self.probability:
                solution_set.add(member)
                
        
        self.solution_set = solution_set
        
        return self.solution_set






    




