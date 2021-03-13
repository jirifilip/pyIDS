import os
import re
import pandas as pd
import logging

from pyarc.qcba.data_structures import QuantitativeDataFrame

from pyids.algorithms.classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.algorithms.ids_multiclass import IDSOneVsAll

logging.basicConfig(level=logging.DEBUG)

directory = "c:/code/python/machine_learning/assoc_rules"
train_directory = os.path.join(directory, "train")
test_directory = os.path.join(directory, "../../../test")


train_files = os.listdir(train_directory)


accuracy_dict = dict()
acc_score_list = []
auc_score_list = []

for train_file in train_files[:20]:
    for algorithm in ["DUSM", "RUSM"]:
        print(train_file)

        rule_count = 100

        acc_score_dict = dict()
        acc_score_dict["file"] = train_file
        acc_score_dict["rule_count"] = rule_count
        acc_score_dict["algorithm"] = algorithm

        auc_score_dict = dict(acc_score_dict)

        df_train = pd.read_csv(os.path.join(train_directory, train_file))
        quant_df_train = QuantitativeDataFrame(df_train)

        df_test = pd.read_csv(os.path.join(test_directory, train_file))
        quant_df_test = QuantitativeDataFrame(df_test)

        cars = mine_CARs(df_train, rule_count)

        ids = IDS(algorithm=algorithm)
        ids.fit(dataframe=quant_df_train, rules=cars)

        acc = ids.score(quant_df_test)

        accuracy_dict[train_file] = acc

        acc_score_dict["accuracy"] = acc


        """
        print("training multi")
        ids_multi = IDSOneVsAll(algorithm=algorithm)
        ids_multi.fit(quant_dataframe=quant_df_train, rule_cutoff=rule_count)

        auc = ids_multi.score_auc(quant_df_test)

        auc_score_dict["auc"] = auc

        print(auc_score_dict)
        """
        print(acc_score_dict)



