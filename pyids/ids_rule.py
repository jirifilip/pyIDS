from pyarc.qcba.data_structures import QuantitativeDataFrame

import numpy as np


class IDSRule:

    DUMMY_LABEL = "N/A"
    
    def __init__(self, class_association_rule):
        self.car = class_association_rule
        self.cover_cache = dict(
            cover=None,
            correct_cover=None,
            incorrect_cover=None,
            rule_cover=None
        )
        self.cache_prepared = False
    
    def __repr__(self):
        return "IDS-" + repr(self.car)

    def __len__(self):
        return len(self.car.antecedent)

    def __hash__(self):
        return hash(self.car)
    

    def calculate_cover(self, quant_dataframe):
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

        rule_cover = self._rule_cover(quant_dataframe)

        class_column_cover = quant_dataframe.dataframe.iloc[:,-1].values != self.car.consequent.value

        return np.logical_and(rule_cover, class_column_cover)