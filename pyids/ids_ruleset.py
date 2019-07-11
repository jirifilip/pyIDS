from .ids_rule import IDSRule

class IDSRuleSet:

    def __init__(self, rules=None):
        self.ruleset = set(rules)


    def __len__(self):
        return len(self.ruleset)


    def sum_rule_length(self):
        rule_lens = []

        for rule in self.ruleset:
            rule_lens.append(len(rule))


        return sum(rule_lens)
    
    def max_rule_length(self):
        rule_lens = []

        for rule in self.ruleset:
            rule_lens.append(len(rule))


        if not rule_lens:
            return 0

        return max(rule_lens)


    @staticmethod
    def from_cba_rules(clazz, cba_rules):
        ids_rules = list(map(IDSRule, cba_rules))
        ids_ruleset = clazz(ids_rules)
        
        return ids_ruleset

