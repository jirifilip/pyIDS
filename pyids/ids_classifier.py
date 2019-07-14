from pyarc.qcba.data_structures import QuantitativeDataFrame
from pyarc.data_structures import TransactionDB
from pyarc.algorithms import top_rules, createCARs

from .ids_rule import IDSRule
from .ids_ruleset import IDSRuleSet
from .ids_objective_function import IDSObjectiveFunction, ObjectiveFunctionParameters
from .ids_optimizer import SLSOptimizer

from .model_selection import encode_label

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from .model_selection import mode
import scipy

import numpy as np
import random

class IDSClassifier:
    
    def __init__(self, rules):
        self.rules = rules


    def get_prediction_rules(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        
        Y = quant_dataframe.dataframe.iloc[:,-1]
        y_pred_dict = dict()
        rules_f1 = dict()

        for rule in self.rules:

            conf = rule.car.confidence
            sup = rule.car.support
            
            y_pred_per_rule = rule.predict(quant_dataframe)
            rule_f1_score = scipy.stats.hmean([conf, sup])

            y_pred_dict.update({rule_f1_score: y_pred_per_rule})
            rules_f1.update({rule_f1_score: rule})

            
        # rules in rows, instances in columns
        y_pred_array = np.array(list(y_pred_dict.values()))

        y_pred = []

        minority_classes = []

        if y_pred_dict:
            for i in range(len(Y)):
                all_NA = np.all(y_pred_array[:,i] == IDSRule.DUMMY_LABEL)
                if all_NA:
                    minority_classes.append(Y[i])

            # if the ruleset covers all instances                     
            default_class = len(Y == Y[0]) / len(Y)
            default_class_label = Y[0]

            if minority_classes:
                default_class = len(Y == mode(minority_classes)) / len(Y)
                default_class_label = mode(minority_classes)

            for i in range(len(Y)):
                y_pred_array_datacase = y_pred_array[:,i]
                non_na_mask = y_pred_array_datacase != IDSRule.DUMMY_LABEL
                
                y_pred_array_datacase_non_na = np.where(non_na_mask)[0]
                print(y_pred_array_datacase_non_na)
                
                if len(y_pred_array_datacase_non_na) > 0:
                    rule_index = y_pred_array_datacase_non_na[0]
                    rule = self.rules[rule_index]

                    y_pred.append((rule.car.confidence, rule.car.consequent.value))
                else:
                    y_pred.append((default_class, default_class_label))

            return y_pred

        else:
            y_pred = len(Y) * [np.inf]

            return y_pred


    def predict(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        
        Y = quant_dataframe.dataframe.iloc[:,-1]
        y_pred_dict = dict()

        for rule in self.rules:

            conf = rule.car.confidence
            sup = rule.car.support
            
            y_pred_per_rule = rule.predict(quant_dataframe)
            rule_f1_score = scipy.stats.hmean([conf, sup])

            y_pred_dict.update({rule_f1_score: y_pred_per_rule})

            
        # rules in rows, instances in columns
        y_pred_array = np.array(list(y_pred_dict.values()))

        y_pred = []

        minority_classes = []

        if y_pred_dict:
            for i in range(len(Y)):
                all_NA = np.all(y_pred_array[:,i] == IDSRule.DUMMY_LABEL)
                if all_NA:
                    minority_classes.append(Y[i])

            # if the ruleset covers all instances                     
            default_class = Y[0]

            if minority_classes:
                default_class = mode(minority_classes)

            for i in range(len(Y)):
                y_pred_array_datacase = y_pred_array[:,i]
                non_na_mask = y_pred_array_datacase != IDSRule.DUMMY_LABEL
                
                y_pred_array_datacase_non_na = y_pred_array_datacase[non_na_mask]
                
                if len(y_pred_array_datacase_non_na) > 0:
                    y_pred.append(y_pred_array_datacase_non_na[0])
                else:
                    y_pred.append(default_class)

            return y_pred

        else:
            y_pred = len(Y) * [mode(Y)]

            return y_pred




class IDS:

    def __init__(self):
        self.clf = None
        self.cacher = None
        self.ids_ruleset = None
    

    def fit(self, quant_dataframe, class_association_rules = None, lambda_array=7*[1], debug=True, objective_scale_factor=1):
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

        optimizer = SLSOptimizer(objective_function, params, debug=debug)

        solution_set = optimizer.optimize()

        self.clf = IDSClassifier(solution_set)

        return self


    def predict(self, quant_dataframe):
        return self.clf.predict(quant_dataframe)

    def get_prediction_rules(self, quant_dataframe):
        return self.clf.get_prediction_rules(quant_dataframe)


    def score(self, quant_dataframe, metric=accuracy_score):
        pred = self.predict(quant_dataframe)
        actual = quant_dataframe.dataframe.iloc[:,-1].values

        return metric(pred, actual)


    def score_auc(self, quant_dataframe):
        pred = self.predict(quant_dataframe)
        actual = quant_dataframe.dataframe.iloc[:,-1].values

        actual, pred = encode_label(actual, pred)

        return roc_auc_score(actual, pred)


class IDSOneVsAll:

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

            self.ids_classifiers.update({class_ : dict(
                quant_dataframe=QuantitativeDataFrame(dataframe_restricted),
                clf=ids_class_clf
            )})


    def fit(self, quant_dataframe, cars=None, rule_cutoff=30, class_name=None, debug=False):

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

            clf.fit(quant_dataframe, cars[:rule_cutoff], debug=debug)


    def _prepare_data_sample(self, quant_dataframe):
        pandas_dataframe = quant_dataframe.dataframe
        ids_classifiers = dict()

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



def mine_CARs(df, rule_cutoff, sample=False):
    txns = TransactionDB.from_DataFrame(df)
    rules = top_rules(txns.string_representation, appearance=txns.appeardict)
    cars = createCARs(rules)

    cars_subset = cars[:rule_cutoff]

    if sample:
        cars_subset = random.sample(cars, rule_cutoff)

    return cars_subset


def mine_IDS_ruleset(df, rule_cutoff):
    cars_subset = mine_CARs(rule_cutoff)
    ids_rls_subset = map(IDSRule, cars_subset)
    ids_ruleset = IDSRuleSet(ids_rls_subset)

    return ids_ruleset