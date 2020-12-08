# pyIDS

pyIDS is a custom implementation of IDS (Interpretable Decision Sets) algorithm introduced in

```LAKKARAJU, Himabindu; BACH, Stephen H.; LESKOVEC, Jure. Interpretable decision sets: A joint framework for description and prediction. In: Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM, 2016. p. 1675-1684.```

If you find this package useful in your research, please cite our paper on this [Interpretable Decision Sets Implementation](https://nb.vse.cz/~klit01/papers/RuleML_Challenge_IDS.pdf):

    Jiri Filip, Tomas Kliegr. PyIDS - Python Implementation of Interpretable Decision Sets Algorithm by Lakkaraju et al, 2016. RuleML+RR2019@Rule Challenge 2019. http://ceur-ws.org/Vol-2438/paper8.pdf


# Installation

The `pyarc`, `pandas`, `scipy` and `numpy` packages need to be installed before using pyIDS.

All of these packages can be installed using `pip`.

For [`pyarc`](https://github.com/jirifilip/pyARC), please refer to the *Installation* section of its README file.

# Examples

training a simple IDS model

```python
import pandas as pd
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS

from pyarc.qcba.data_structures import QuantitativeDataFrame


df = pd.read_csv("./data/iris0.csv")
cars = mine_CARs(df, rule_cutoff=50)
lambda_array = [1, 1, 1, 1, 1, 1, 1]

quant_dataframe = QuantitativeDataFrame(df)

ids = IDS(algorithm="SLS")
ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

acc = ids.score(quant_dataframe)
```

optimizing for best lambda parameters using coordinate ascent, as described in the original paper

```python
import pandas as pd

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent

from pyarc.qcba.data_structures import QuantitativeDataFrame


df = pd.read_csv("./data/titanic.csv")
quant_df = QuantitativeDataFrame(df)
cars = mine_CARs(df, 20)


def fmax(lambda_dict):
    print(lambda_dict)
    ids = IDS(algorithm="SLS")
    ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=list(lambda_dict.values()))
    auc = ids.score_auc(quant_df)
    print(auc)
    return auc



coord_asc = CoordinateAscent(
    func=fmax,
    func_args_ranges=dict(
        l1=(1, 1000),
        l2=(1, 1000),
        l3=(1, 1000),
        l4=(1, 1000),
        l5=(1, 1000),
        l6=(1, 1000),
        l7=(1, 1000)
    ),
    ternary_search_precision=50,
    max_iterations=3
)

best_lambdas = coord_asc.fit()
```

