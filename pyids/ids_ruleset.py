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


    def fraction_overlap(self):
        pass


    def fraction_uncovered(self):
        pass

    def average_rule_length(self):
        pass

    def fraction_of_classes(self):
        pass
