import numpy as np
import logging


class IDSCacher:

    def __init__(self):
        self.overlap_cache = dict()

        self.logger = logging.Logger(IDSCacher.__name__)

    def overlap(self, rule1, rule2):
        return self.overlap_cache[repr(rule1) + repr(rule2)]

    def calculate_overlap(self, all_rules, quant_dataframe):
        for rule in all_rules.ruleset:
            rule.calculate_cover(quant_dataframe)

        self.logger.debug("cover cache prepared")

        for rule_i in all_rules.ruleset:
            for rule_j in all_rules.ruleset:
                
                overlap_tmp = rule_i.rule_overlap(rule_j, quant_dataframe)
                overlap_len = np.sum(overlap_tmp)

                self.overlap_cache[repr(rule_i) + repr(rule_j)] = overlap_len
        
        self.logger.debug("overlap cache prepared")
    


