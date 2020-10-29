import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from pyarc.qcba.data_structures import QuantitativeDataFrame

from ..ids import IDS

class IDSVisualization:

    def __init__(self, ids_clf, quant_dataframe):
        if type(quant_dataframe) != QuantitativeDataFrame:
            raise Exception("type of quant_dataframe must be QuantitativeDataFrame")

        if type(ids_clf) != IDS:
            raise Exception("type of ids_clf must by IDS")

        self.pd_dataframe = quant_dataframe.dataframe
        self.colnames = list(self.pd_dataframe.columns)
        self.colnames_len = len(self.colnames)
        self.colnames_x = self.colnames[:self.colnames_len - 1]
        self.colnames_y = self.colnames[self.colnames_len - 1] 

        self.colnames_x_combinations = list(itertools.combinations(self.colnames_x, 2))


    def visualize_dataframe(self, figsize):
        fig, axes = plt.subplots(self.colnames_len)
        fig.set_size_inches(*figsize)

        for idx, ax in enumerate(axes):
            col_x, col_y = self.colnames_x_combinations[idx]
            
            x = self.pd_dataframe[col_x]
            y = self.pd_dataframe[col_y]
            color = self.pd_dataframe[self.colnames_y].values
            
            ax.scatter(x, y, c=color)

