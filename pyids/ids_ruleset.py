from .ids_rule import IDSRule

import xml.etree.ElementTree as ET

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


    @classmethod
    def from_cba_rules(clazz, cba_rules):
        ids_rules = list(map(IDSRule, cba_rules))
        ids_ruleset = clazz(ids_rules)
        
        return ids_ruleset


    def to_dict(self):
        rule_dict_list = []

        for rule in self.ruleset:
            rule_dict_list.append(rule.to_dict())

        return rule_dict_list


    def to_xml(self, return_ET=True):
        rule_xml_list = ET.Element("rules")

        for rule in self.ruleset:
            rule_xml = rule.to_xml()

            rule_xml_list.append(rule_xml)

        if return_ET:
            return ET.ElementTree(rule_xml_list)
        else:
            return ET.dump(rule_xml_list)




