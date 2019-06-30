import numpy as np

class IDSCacher:

    def __init__(self, debug=True):
        self.overlap_cache = dict()

        self.debug = debug


    def overlap(self, rule1, rule2):
        return self.overlap_cache[repr(rule1) + repr(rule2)]


    
    def calculate_overlap(self, all_rules, quant_dataframe):
        for rule in all_rules.ruleset:
            rule.calculate_cover(quant_dataframe)
        print("cover cache prepared")


        len_all_rules = len(all_rules)
        progress_bars = 20
        progress_bar_step = len_all_rules / progress_bars
        progress_bar_curr = 1        


        for i, rule_i in enumerate(all_rules.ruleset):
            for j, rule_j in enumerate(all_rules.ruleset):
                
                overlap_tmp = rule_i.rule_overlap(rule_j, quant_dataframe)
                overlap_len = np.sum(overlap_tmp)

                self.overlap_cache[repr(rule_i) + repr(rule_j)] = overlap_len
        
        print("overlap cache prepared")
    


