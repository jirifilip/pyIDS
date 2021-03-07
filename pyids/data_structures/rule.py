from xml.etree import ElementTree as ET

from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import ClassAssocationRule

from sklearn.metrics import f1_score
import numpy as np
import xml.etree.ElementTree as ET


class IDSRule:

    DUMMY_LABEL = "N/A"
    
    def __init__(self, class_association_rule: ClassAssocationRule):
        self.car = class_association_rule
        self.cover_cache = dict(
            cover=None,
            correct_cover=None,
            incorrect_cover=None,
            rule_cover=None
        )
        self.cache_prepared = False
        self.f1 = 0

    def calc_f1(self, quant_dataframe: QuantitativeDataFrame):
        ground_truth = quant_dataframe.dataframe.iloc[:, -1]
        predictions = self.predict(quant_dataframe)

        f1 = f1_score(ground_truth, predictions, average="micro")

        return f1

    def predict(self, quant_dataframe: QuantitativeDataFrame):
        correct_cover_mask = self.correct_cover(quant_dataframe)

        predictions = np.where(correct_cover_mask, self.car.consequent.value, "DUMMY_LABEL")

        return predictions

    def __repr__(self):
        args = [
            self.car.antecedent.string(),
            "{" + self.car.consequent.string() + "}",
            self.car.support,
            self.car.confidence,
            self.f1,
            self.car.rulelen,
            self.car.rid
        ]

        text = "IDSRule {} => {} sup: {:.2f} conf: {:.2f}, f1: {:.2f}, len: {}, id: {}".format(*args)

        return text

    def __len__(self):
        return len(self.car.antecedent)

    def __hash__(self):
        return hash(self.car)

    def to_dict(self):
        rule_dict = dict(antecedent=[], consequent={})

        for label, value in self.car.antecedent:
            rule_dict["antecedent"].append(dict(name=label, value=value))

        label, value = self.car.consequent

        rule_dict["consequent"].update(dict(name=label, value=value))

        return rule_dict

    def to_ruleml_xml(self):
        rule_dict = self.to_dict()

        rule = ET.Element("Implies")

        consequent = ET.SubElement(rule, "head")

        label_element = ET.SubElement(consequent, "Atom")
        var_element = ET.SubElement(label_element, "Var")
        var_element.text = rule_dict["consequent"]["name"]

        rel_element = ET.SubElement(label_element, "Rel")
        rel_element.text = rule_dict["consequent"]["value"]

        antecedent = ET.SubElement(rule, "body")

        for antecedent_member in rule_dict["antecedent"]:
            for label, value in antecedent_member.items():
                label_element = ET.SubElement(antecedent, "Atom")
                var_element = ET.SubElement(label_element, "Var")
                var_element.text = label

                rel_element = ET.SubElement(label_element, "Rel")
                rel_element.text = value

        return rule

    def to_xml(self):
        rule_dict = self.to_dict()

        rule = ET.Element("Rule")
        antecedent = ET.SubElement(rule, "Antecedent")

        for antecedent_member in rule_dict["antecedent"]:
            for label, value in antecedent_member.items():
                label_element = ET.SubElement(antecedent, label)
                label_element.text = value

        consequent = ET.SubElement(rule, "Consequent")

        for label, value in rule_dict["consequent"].items():
            label_element = ET.SubElement(consequent, label)
            label_element.text = value

        return rule

    def calculate_cover(self, quant_dataframe: QuantitativeDataFrame):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        self.cover_cache["cover"] = self._cover(quant_dataframe)
        self.cover_cache["correct_cover"] = self._correct_cover(quant_dataframe)
        self.cover_cache["incorrect_cover"] = self._incorrect_cover(quant_dataframe)
        self.cover_cache["rule_cover"] = self._rule_cover(quant_dataframe)

        self.cover_cache["cover_len"] = np.sum(self.cover_cache["cover"])
        self.cover_cache["correct_cover_len"] = np.sum(self.cover_cache["correct_cover"])
        self.cover_cache["incorrect_cover_len"] = np.sum(self.cover_cache["incorrect_cover"])
        self.cover_cache["rule_cover_len"] = np.sum(self.cover_cache["rule_cover"])

        self.cache_prepared = True

        self.f1 = self.calc_f1(quant_dataframe)

    def cover(self, quant_dataframe):
        if not self.cache_prepared:
            raise Exception("Caches not prepared yet")

        return self.cover_cache["cover"]

    def correct_cover(self, quant_dataframe):
        if not self.cache_prepared:
            raise Exception("Caches not prepared yet")

        return self.cover_cache["correct_cover"]

    def incorrect_cover(self, quant_dataframe):
        if not self.cache_prepared:
            raise Exception("Caches not prepared yet")

        return self.cover_cache["incorrect_cover"]

    def rule_cover(self, quant_dataframe):
        if not self.cache_prepared:
            raise Exception("Caches not prepared yet")

        return self.cover_cache["rule_cover"]

    def _cover(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        cover, _ = quant_dataframe.find_covered_by_rule_mask(self.car)

        return cover

    def rule_overlap(self, other, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame") 

        if type(other) != IDSRule:
            raise Exception("Type of other must be IDSRule")

        cover1 = self.cover(quant_dataframe)
        cover2 = other.cover(quant_dataframe)

        overlap = np.logical_and(cover1, cover2)

        return overlap

    def predict(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        cover = self.cover(quant_dataframe)

        class_label = self.car.consequent.value

        prediction = np.where(cover, class_label, IDSRule.DUMMY_LABEL)

        return prediction

    def _rule_cover(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        cover_antecedent, cover_consequent = quant_dataframe.find_covered_by_rule_mask(self.car)

        rule_cover = cover_antecedent & cover_consequent

        return rule_cover

    def _correct_cover(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        rule_cover = self._rule_cover(quant_dataframe)

        class_column_cover = quant_dataframe.dataframe.iloc[:,-1].values == self.car.consequent.value

        return np.logical_and(rule_cover, class_column_cover)

    def _incorrect_cover(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        correct_cover = self._correct_cover(quant_dataframe)

        return np.logical_not(correct_cover)

    def __gt__(self, other):
        """
        precedence operator. Determines if this rule
        has higher precedence. Rules are sorted according
        to their f1 score.
        """

        f1_score_self = self.f1
        f1_score_other = other.f1

        return f1_score_self > f1_score_other

    def __lt__(self, other):
        """
        rule precedence operator
        """
        return not self > other


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