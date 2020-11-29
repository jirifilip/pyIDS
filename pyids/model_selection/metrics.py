from ..data_structures.ids_ruleset import IDSRuleSet
from pyarc.qcba.data_structures import QuantitativeDataFrame

import numpy as np


def fraction_overlap(ids_model, quant_dataframe):
    ruleset = IDSRuleSet(ids_model.clf.rules)

    if type(quant_dataframe) != QuantitativeDataFrame:
        raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

    for rule in ruleset.ruleset:
        rule.calculate_cover(quant_dataframe)

    ruleset_len = len(ruleset)
    dataset_len = quant_dataframe.dataframe.index.size

    denominator = ruleset_len * (ruleset_len - 1)

    if denominator == 0 or dataset_len == 0:
        return 0

    overlap_sum = 0
    for i, rule_i in enumerate(ruleset.ruleset):
        for j, rule_j in enumerate(ruleset.ruleset):
            if i <= j:
                continue

            overlap_sum += np.sum(rule_i.rule_overlap(rule_j, quant_dataframe)) / dataset_len

    frac_overlap = 2 / denominator * overlap_sum

    return frac_overlap


def fraction_uncovered(ids_model, quant_dataframe):
    ruleset = IDSRuleSet(ids_model.clf.rules)

    if type(quant_dataframe) != QuantitativeDataFrame:
        raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")


    dataset_len = quant_dataframe.dataframe.index.size
    cover_cummulative_mask = np.zeros(dataset_len)

    for rule in ruleset.ruleset:
        cover = rule._cover(quant_dataframe)
        cover_cummulative_mask = np.logical_or(cover_cummulative_mask, cover)

    if dataset_len == 0:
        return 0

    frac_uncovered = 1 - 1 / dataset_len * np.sum(cover_cummulative_mask)

    return frac_uncovered


def average_rule_width(ids_model):
    ruleset = IDSRuleSet(ids_model.clf.rules)

    rule_widths = []

    for rule in ruleset.ruleset:
        rule_widths.append(len(rule))

    if rule_widths:
        avg_rule_width = np.mean(rule_widths)
    else:
        avg_rule_width = 0

    return avg_rule_width


def fraction_classes(ids_model, quant_dataframe):
    ruleset = IDSRuleSet(ids_model.clf.rules)

    if type(quant_dataframe) != QuantitativeDataFrame:
        raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

    dataset_classes = set(quant_dataframe.dataframe.iloc[:,-1].values)
    rules_covered_classes = set()

    if len(dataset_classes) == 0:
        return 0

    for rule in ruleset.ruleset:
        covered_class = rule.car.consequent.value
        rules_covered_classes.add(covered_class)

    rules_covered_classes.add(ids_model.clf.default_class)

    frac_classes = len(rules_covered_classes) / len(dataset_classes)

    return frac_classes


def calculate_ruleset_statistics(ids_model, quant_dataframe):
    result = dict(
        fraction_overlap=fraction_overlap(ids_model, quant_dataframe),
        fraction_classes=fraction_classes(ids_model, quant_dataframe),
        fraction_uncovered=fraction_uncovered(ids_model, quant_dataframe),
        average_rule_width=average_rule_width(ids_model),
        ruleset_length=len(ids_model.clf.rules)
    )

    return result


def calculate_metrics_average(metrics_list: list):
    metrics_avg = dict(
        fraction_overlap=np.mean([metrics["fraction_overlap"] for metrics in metrics_list]),
        fraction_classes=np.mean([metrics["fraction_classes"] for metrics in metrics_list]),
        fraction_uncovered=np.mean([metrics["fraction_uncovered"] for metrics in metrics_list]),
        average_rule_width=np.mean([metrics["average_rule_width"] for metrics in metrics_list]),
        ruleset_length=np.mean([metrics["ruleset_length"] for metrics in metrics_list])
    )

    return metrics_avg