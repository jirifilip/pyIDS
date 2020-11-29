from pyarc.qcba.data_structures import QuantitativeDataFrame
from ..model_selection import calculate_metrics_average

from .ids_classifier import mine_CARs
from .ids import IDS

import numpy as np
import pandas as pd
import logging


class IDSOneVsAll:

    def __init__(self, algorithm: str = "SLS"):
        self.algorithm = algorithm

        self.quant_dataframe = None
        self.pandas_dataframe = None
        self.ids_classifiers = dict()

        self.class_name = None
        self.other_class_label = "OTHER"

        self.logger = logging.getLogger(IDSOneVsAll.__name__)

    def _prepare(self, quant_dataframe: QuantitativeDataFrame, class_name: str):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        self.quant_dataframe = quant_dataframe
        self.pandas_dataframe = self.quant_dataframe.dataframe

        self.class_name = class_name if class_name else self.pandas_dataframe.columns[-1]
        class_column = self.pandas_dataframe[self.class_name]
        unique_classes = np.unique(class_column.values)

        if len(unique_classes) < 3:
            raise Exception("Number of distinct classes must be greater than 2, otherwise use binary classifier")

        for class_ in unique_classes:
            dataframe_restricted = self.pandas_dataframe.copy()
            dataframe_class_column_restricted = np.where(class_column == class_, class_, self.other_class_label)

            dataframe_restricted[self.class_name] = dataframe_class_column_restricted

            ids_class_clf = IDS(algorithm=self.algorithm)

            self.ids_classifiers.update({class_: dict(
                quant_dataframe=QuantitativeDataFrame(dataframe_restricted),
                rules=None,
                clf=ids_class_clf
            )})

    def mine_rules(self, quant_dataframe: QuantitativeDataFrame, rule_cutoff: int = 30, class_name: str = None):
        self._prepare(quant_dataframe, class_name)

        for class_, clf_dict in self.ids_classifiers.items():
            self.logger.debug(f"Mining rules for class: {class_}")

            quant_dataframe = clf_dict["quant_dataframe"]
            pandas_dataframe = quant_dataframe.dataframe

            rules = mine_CARs(pandas_dataframe, rule_cutoff=rule_cutoff)

            self.logger.debug(f"# of used rules rules: {len(rules)}")

            clf_dict["rules"] = rules

    def fit(self, quant_dataframe: QuantitativeDataFrame, rule_cutoff=30, lambda_array=7 * [1], class_name=None):
        self.mine_rules(quant_dataframe, rule_cutoff, class_name)

        for class_, clf_dict in self.ids_classifiers.items():
            self.logger.debug(f"Training classifier for class: {class_}")

            clf = clf_dict["clf"]
            rules = clf_dict["rules"]
            quant_dataframe_for_class = clf_dict["quant_dataframe"]

            clf.fit(quant_dataframe_for_class, rules, lambda_array=lambda_array)

            self.logger.debug(f"Default class {clf.clf.default_class}")

    def score_interpretability_metrics(self, quant_dataframe: QuantitativeDataFrame):
        quant_dataframe_split = self.split_data_by_class(quant_dataframe)

        interpretability_metrics_all = []

        for class_name, clf_dict in self.ids_classifiers.items():
            clf: IDS = clf_dict["clf"]

            if class_name not in quant_dataframe_split.keys():
                continue

            quant_df = quant_dataframe_split[class_name]

            metrics = clf.score_interpretability_metrics(quant_df)
            interpretability_metrics_all.append(metrics)

        return calculate_metrics_average(interpretability_metrics_all)

    def summary(self):
        summary_list = []

        for class_name, clf_dict in self.ids_classifiers.items():
            clf: IDS = clf_dict["clf"]
            quant_df = clf_dict["quant_dataframe"]
            rules = clf_dict["rules"]

            basic_metrics = dict(
                class_name=class_name,
                n_data=len(quant_df.dataframe),
                n_mined_rules=len(rules),
                model_accuracy=clf.score(quant_df),
                model_AUC=clf.score_auc(quant_df),
                default_class=clf.clf.default_class,
                default_class_confidence=clf.clf.default_class_confidence
            )

            basic_metrics.update(clf.score_interpretability_metrics(quant_df))

            summary_list.append(basic_metrics)

        summary_df = pd.DataFrame(summary_list)

        return summary_df

    def split_data_by_class(self, quant_dataframe):
        pandas_dataframe = quant_dataframe.dataframe

        class_column = pandas_dataframe[self.class_name] if self.class_name else pandas_dataframe.iloc[:, -1]
        unique_classes = np.unique(class_column.values)

        restricted_quant_dataframes = dict()

        for class_ in unique_classes:
            dataframe_restricted = pandas_dataframe.copy()
            dataframe_class_column_restricted = np.where(class_column == class_, class_, self.other_class_label)

            if self.class_name:
                dataframe_restricted[self.class_name] = dataframe_class_column_restricted
            else:
                dataframe_restricted.iloc[:, -1] = dataframe_class_column_restricted

            dataframe = QuantitativeDataFrame(dataframe_restricted)
            restricted_quant_dataframes[class_] = dataframe

        return restricted_quant_dataframes

    def score_auc(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("type of quant_dataframe must be QuantitativeDataFrame")

        AUCs = []

        restricted_quant_dataframes = self.split_data_by_class(quant_dataframe)

        for class_, clf_dict in self.ids_classifiers.items():
            self.logger.debug(f"scoring class: {class_}")

            clf = clf_dict["clf"]

            dataframe_test = restricted_quant_dataframes[class_]

            auc = clf.score_auc(dataframe_test)

            AUCs.append(auc)

        auc_mean = np.mean(AUCs)

        return auc_mean