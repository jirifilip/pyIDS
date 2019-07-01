from ..ids_ruleset import IDSRuleSet
from pyarc.qcba.data_structures import QuantitativeDataFrame

import numpy as np

def fraction_overlap(ruleset, quant_dataframe):
    if type(ruleset) != IDSRuleSet:
        raise Exception("Type of ruleset must be IDSRuleSet")

    if type(quant_dataframe) != QuantitativeDataFrame:
        raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

    for rule in ruleset.rules:
        rule.calculate_cover(quant_dataframe)

    ruleset_len = len(ruleset)
    dataset_len = quant_dataframe.dataframe.index.size

    overlap_sum = 0
    for i, rule_i in enumerate(ruleset.rules):
        for j, rule_j in enumerate(ruleset.rules):

            if i <= j:
                continue

            overlap_sum += len(rule_i.rule_overlap(rule_j)) / dataset_len


    frac_overlap = 2 / (ruleset_len * (ruleset_len - 1)) * overlap_sum

    return frac_overlap


def fraction_uncovered(ruleset, quant_dataframe):
    if type(ruleset) != IDSRuleSet:
        raise Exception("Type of ruleset must be IDSRuleSet")

    if type(quant_dataframe) != QuantitativeDataFrame:
        raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")


    dataset_len = quant_dataframe.dataframe.index.size
    cover_cummulative_mask = np.zeros(dataset_len)

    for rule in ruleset.rules:
        cover = rule._cover(quant_dataframe)
        cover_cummulative_mask = cover_cummulative_mask | cover
    

    frac_uncovered = 1 - 1 / dataset_len * np.sum(cover_cummulative_mask)

    return frac_uncovered


def average_rule_width(ruleset):
    if type(ruleset) != IDSRuleSet:
        raise Exception("Type of ruleset must be IDSRuleSet")


    rule_widths = []

    for rule in ruleset.rules:
        rule_widths.append(len(rule))
    

    avg_rule_width = np.mean(rule_widths)

    return avg_rule_width


def fraction_classes(ruleset, quant_dataframe):
    if type(ruleset) != IDSRuleSet:
        raise Exception("Type of ruleset must be IDSRuleSet")

    if type(quant_dataframe) != QuantitativeDataFrame:
        raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")


    dataset_classes = set(quant_dataframe.dataframe.iloc[:,-1].values)
    rules_covered_classes = set()

    for rule in ruleset.rules:
        covered_class = rule.car.consequent.values
        rules_covered_classes.add(covered_class)


    frac_classes = len(rules_covered_classes) / len(dataset_classes)

    return frac_classes
