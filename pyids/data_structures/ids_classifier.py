from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import top_rules, createCARs

from .ids_rule import IDSRule
from .ids_ruleset import IDSRuleSet
from .ids_objective_function import IDSObjectiveFunction, ObjectiveFunctionParameters
from .ids_optimizer import SLSOptimizer, DLSOptimizer, DeterministicUSMOptimizer, RandomizedUSMOptimizer

from ..model_selection import encode_label, calculate_ruleset_statistics

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from ..model_selection import mode
import scipy

import numpy as np
import random
import copy

class IDSClassifier:
    
    def __init__(self, rules):
        self.rules = rules
        self.default_class = None
        self.default_class_confidence = None
        self.quant_dataframe_train = None

        
    def calculate_default_class(self):
        predicted_classes = self.predict(self.quant_dataframe_train)
        not_classified_idxes = [ idx for idx, val in enumerate(predicted_classes) if val == None ]
        classes = self.quant_dataframe_train.dataframe.iloc[:, -1]

        actual_classes = list(self.quant_dataframe_train.dataframe.iloc[not_classified_idxes, -1])

        # return random class
        if not list(actual_classes):
            self.default_class = random.sample(list(np.unique(classes)), 1)[0]
            self.default_class_confidence = 1

        else:
            minority_class = mode(actual_classes)

            self.default_class = minority_class
            self.default_class_confidence = actual_classes.count(minority_class) / len(actual_classes)


    def predict(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        predicted_classes = []
    
        for _, row in quant_dataframe.dataframe.iterrows():
            appended = False
            for rule in self.rules:
                antecedent_dict = dict(rule.car.antecedent)  
                counter = True

                for name, value in row.iteritems():
                    if name in antecedent_dict:
                        rule_value = antecedent_dict[name]

                        counter &= rule_value == value

                if counter:
                    _, predicted_class = rule.car.consequent
                    predicted_classes.append(predicted_class)

                    appended = True

                    break

                
            if not appended:
                predicted_classes.append(self.default_class)

                    
        return predicted_classes


class IDS:

    def __init__(self, algorithm="SLS"):
        self.clf = None
        self.cacher = None
        self.ids_ruleset = None
        self.algorithms = dict(
            SLS=SLSOptimizer,
            DLS=DLSOptimizer,
            DUSM=DeterministicUSMOptimizer,
            RUSM=RandomizedUSMOptimizer
        )

        self.algorithm = algorithm
    

    def fit(self, quant_dataframe, class_association_rules=None, lambda_array=7*[1], default_class="majority_class_in_uncovered", debug=True, objective_scale_factor=1, random_seed=None):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")


        # init params
        params = ObjectiveFunctionParameters()
        
        if not self.ids_ruleset:
            ids_rules = list(map(IDSRule, class_association_rules))
            all_rules = IDSRuleSet(ids_rules)
            params.params["all_rules"] = all_rules
        elif self.ids_ruleset and not class_association_rules:
            print("using provided ids ruleset and not class association rules")
            params.params["all_rules"] = self.ids_ruleset
        
        params.params["len_all_rules"] = len(params.params["all_rules"])
        params.params["quant_dataframe"] = quant_dataframe
        params.params["lambda_array"] = lambda_array
        
        # objective function
        objective_function = IDSObjectiveFunction(objective_func_params=params, cacher=self.cacher, scale_factor=objective_scale_factor)

        optimizer = self.algorithms[self.algorithm](objective_function, params, debug=debug, random_seed=random_seed)

        solution_set = optimizer.optimize()

        self.clf = IDSClassifier(solution_set)
        self.clf.rules = sorted(self.clf.rules, reverse=True)
        self.clf.quant_dataframe_train = quant_dataframe

        if default_class == "majority_class_in_all":
            classes = quant_dataframe.dataframe.iloc[:, -1]
            self.clf.default_class = mode(classes)
            self.clf.default_class_confidence = classes.count(self.clf.default_class) / len(classes)
        elif default_class == "majority_class_in_uncovered":
            self.clf.calculate_default_class()

        return self


    def predict(self, quant_dataframe):
        return self.clf.predict(quant_dataframe)

    def get_prediction_rules(self, quant_dataframe):
        return self.clf.get_prediction_rules(quant_dataframe)

    def score(self, quant_dataframe, metric=accuracy_score):
        pred = self.predict(quant_dataframe)
        actual = quant_dataframe.dataframe.iloc[:,-1].values
        return metric(pred, actual)

    def _calculate_auc_for_ruleconf(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        
        confidences = []
    
        for _, row in quant_dataframe.dataframe.iterrows():
            appended = False
            for rule in self.rules:
                antecedent_dict = dict(rule.car.antecedent)  
                counter = True

                for name, value in row.iteritems():
                    if name in antecedent_dict:
                        rule_value = antecedent_dict[name]

                        counter &= rule_value == value

                if counter:
                    confidences.append(rule.car.confidence)

                    appended = True

                    break

                
            if not appended:
                confidences.append(self.clf.default_class_confidence)

                    
        actual_classes = quant_dataframe.dataframe.iloc[:, -1].values
        predicted_classes = self.predict(quant_dataframe)

        actual, pred = encode_label(actual_classes, predicted_classes)

        corrected_confidences = np.where(pred == "1", predicted_classes, 1 - predicted_classes)

        return corrected_confidences


    def _calcutate_auc_classical(self, quant_dataframe):
        pred = self.predict(quant_dataframe)
        actual = quant_dataframe.dataframe.iloc[:,-1].values

        actual, pred = encode_label(actual, pred)

        return roc_auc_score(actual, pred)


    def score_auc(self, quant_dataframe, confidence_based=False):
        if confidence_based:
            return self._calculate_auc_for_ruleconf(quant_dataframe)
        else:
            return self._calcutate_auc_classical(quant_dataframe)


    def score_interpretable_metrics(self, quant_dataframe):
        current_ruleset = IDSRuleSet(self.clf.rules)
        
        stats = calculate_ruleset_statistics(current_ruleset, quant_dataframe)

        return stats


class IDSOneVsAll:

    def __init__(self, clf_blueprint=None):
        self.clf_blueprint = clf_blueprint

    def _prepare(self, quant_dataframe, class_name):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")

        self.quant_dataframe = quant_dataframe   
        self.pandas_dataframe = self.quant_dataframe.dataframe
        self.ids_classifiers = dict()

        self.class_name = class_name
        self.other_class_label = "OTHER"

        class_column = self.pandas_dataframe[class_name] if class_name else self.pandas_dataframe.iloc[:,-1]
        unique_classes = np.unique(class_column.values)

        if len(unique_classes) < 3:
            raise Exception("Number of distinct classes must be greater than 2, otherwise use binary classifier")

        for class_ in unique_classes:
            # TODO
            # find a better way than copy
            dataframe_restricted = self.pandas_dataframe.copy()
            dataframe_class_column_restricted = np.where(class_column == class_, class_, self.other_class_label)

            if class_name:
                dataframe_restricted[class_name] = dataframe_class_column_restricted
            else:
                dataframe_restricted.iloc[:,-1] = dataframe_class_column_restricted
            
            ids_class_clf = IDS()

            if self.clf_blueprint:
                ids_class_clf = copy.copy(self.clf_blueprint)

            self.ids_classifiers.update({class_ : dict(
                quant_dataframe=QuantitativeDataFrame(dataframe_restricted),
                clf=ids_class_clf
            )})


    def fit(self, quant_dataframe, cars=None, rule_cutoff=30, lambda_array=7*[1], class_name=None, debug=False):

        self.quant_dataframe_train = quant_dataframe

        self._prepare(quant_dataframe, class_name)

        for class_, clf_dict in self.ids_classifiers.items():
            print("training class:", class_)

            clf = clf_dict["clf"]
            quant_dataframe = clf_dict["quant_dataframe"]
            pandas_dataframe = quant_dataframe.dataframe

            txns = TransactionDB.from_DataFrame(pandas_dataframe)
            rules = top_rules(txns.string_representation, appearance=txns.appeardict)
            cars = createCARs(rules)
            cars.sort(reverse=True)

            clf.fit(quant_dataframe, cars[:rule_cutoff], lambda_array=lambda_array, debug=debug)


    def _prepare_data_sample(self, quant_dataframe):
        pandas_dataframe = quant_dataframe.dataframe

        class_column = pandas_dataframe[self.class_name] if self.class_name else pandas_dataframe.iloc[:,-1]
        unique_classes = np.unique(class_column.values)

        restricted_quant_dataframes = []

        for class_ in unique_classes:
            # TODO
            # find a better way than copy
            dataframe_restricted = pandas_dataframe.copy()
            dataframe_class_column_restricted = np.where(class_column == class_, class_, self.other_class_label)

            if self.class_name:
                dataframe_restricted[self.class_name] = dataframe_class_column_restricted
            else:
                dataframe_restricted.iloc[:,-1] = dataframe_class_column_restricted
            
            dataframe = QuantitativeDataFrame(dataframe_restricted)
            restricted_quant_dataframes.append(dataframe)

        return restricted_quant_dataframes


    def score_auc(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("type of quant_dataframe must be QuantitativeDataFrame")


        AUCs = []

        restricted_quant_dataframes = self._prepare_data_sample(quant_dataframe)

        for idx, (class_, clf_dict) in enumerate(self.ids_classifiers.items()):
            print("scoring class:", class_)

            clf = clf_dict["clf"]

            dataframe_test = restricted_quant_dataframes[idx]

            auc = clf.score_auc(dataframe_test)

            AUCs.append(auc)


        auc_mean = np.mean(AUCs)

        return auc_mean



def mine_CARs(df, rule_cutoff, sample=False, random_seed=None, **top_rules_kwargs):
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    txns = TransactionDB.from_DataFrame(df)
    rules = top_rules(txns.string_representation, appearance=txns.appeardict, **top_rules_kwargs)
    cars = createCARs(rules)

    cars_subset = cars[:rule_cutoff]

    if sample:
        cars_subset = random.sample(cars, rule_cutoff)

    return cars_subset


def mine_IDS_ruleset(df, rule_cutoff, random_seed=None, **top_rules_kwargs):
    cars_subset = mine_CARs(df, rule_cutoff, random_seed=random_seed, **top_rules_kwargs)
    ids_rls_subset = map(IDSRule, cars_subset)
    ids_ruleset = IDSRuleSet(ids_rls_subset)

    return ids_ruleset