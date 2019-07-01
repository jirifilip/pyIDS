from pyarc.qcba.data_structures import QuantitativeDataFrame

from .ids_rule import IDSRule
from .ids_ruleset import IDSRuleSet
from .ids_objective_function import IDSObjectiveFunction, ObjectiveFunctionParameters
from .ids_optimizer import SLSOptimizer

from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import mode

class IDSClassifier:
    
    def __init__(self, rules):
        self.rules = rules


    def predict(self, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            print("Type of quant_dataframe must be QuantitativeDataFrame")

        Y = quant_dataframe.dataframe.iloc[:,-1]
        y_pred_dict = dict()

        for rule in self.rules:
            y_pred_per_rule = rule.predict(quant_dataframe)

            rule_f1_score = f1_score(Y, y_pred_per_rule, average="micro")

            y_pred_dict.update({rule_f1_score: y_pred_per_rule})


        y_pred = []

        if y_pred_dict:
            top_f1_score = sorted(y_pred_dict.keys(), reverse=True)[0]


            for subscript in range(len(Y)):
                v_list = []
                for k, v in y_pred_dict.items():
                    v_list.append(v[subscript])
                set_v_list = set(v_list)
            
                if list(set_v_list)[0] == IDSRule.DUMMY_LABEL:         # "For data points that satisfy zero itemsets, we predict the majority class label in the training data,"
                    y_pred.append(mode(Y).mode[0])
                elif len(list(set_v_list)) - 1 >  len(set(Y)): # "and for data points that satisfy more than one itemset, we predict using the rule with the highest F1 score on the training data."
                    y_pred.append(y_pred_dict[top_f1_score][subscript])
                else:                                          # unique
                    y_pred.append(list(set(v_list))[0])
        
    
            return y_pred

        else:
            for _ in range(len(Y)):
                y_pred.append(mode(Y).mode[0])

            return y_pred



class IDS:

    def __init__(self):
        self.clf = None
    

    def fit(self, quant_dataframe, class_association_rules, lambda_array=7*[1], debug=True):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("Type of quant_dataframe must be QuantitativeDataFrame")


        ids_rules = list(map(IDSRule, class_association_rules))

        # init params
        params = ObjectiveFunctionParameters()
        all_rules = IDSRuleSet(ids_rules)
        params.params["all_rules"] = all_rules
        params.params["quant_dataframe"] = quant_dataframe
        params.params["lambda_array"] = lambda_array
        
        # objective function
        objective_function = IDSObjectiveFunction(objective_func_params=params)

        optimizer = SLSOptimizer(objective_function, params, debug=debug)

        solution_set = optimizer.optimize()

        self.clf = IDSClassifier(solution_set)

        return self


    def predict(self, quant_dataframe):
        return self.clf.predict(quant_dataframe)


    def score(self, quant_dataframe, metric=accuracy_score):
        pred = self.predict(quant_dataframe)
        actual = quant_dataframe.dataframe.iloc[:,-1].values

        return metric(pred, actual)



    