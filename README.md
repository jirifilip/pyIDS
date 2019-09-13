# pyIDS

pyIDS is a custom implementation of IDS (Interpretable Decision Sets) algorithm introduced in

```LAKKARAJU, Himabindu; BACH, Stephen H.; LESKOVEC, Jure. Interpretable decision sets: A joint framework for description and prediction. In: Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM, 2016. p. 1675-1684.```

If you find this package useful in your research, please cite our paper on this [Interpretable Decision Sets Implementation](https://nb.vse.cz/~klit01/papers/RuleML_Challenge_IDS.pdf):

    Jiri Filip, Tomas Kliegr. PyIDS - Python Implementation of Interpretable Decision Sets Algorithm by Lakkaraju et al, 2016. RuleML+RR2019@Rule Challenge 2019. http://ceur-ws.org/Vol-2438/paper8.pdf



# Examples

training a simple IDS model

```python
import pandas as pd
from pyids.ids_classifier import IDS, mine_CARs

from pyarc.qcba.data_structures import QuantitativeDataFrame


df = pd.read_csv("./data/iris0.csv")
cars = mine_CARs(df, rule_cutoff=50)
lambda_array = [1, 1, 1, 1, 1, 1, 1]

quant_dataframe = QuantitativeDataFrame(df)

ids = IDS()
ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array, debug=False)

acc = ids.score(quant_dataframe)
```

training a One-vs-all IDS model

```python
import pandas as pd
from pyids.ids_classifier import IDSOneVsAll, mine_CARs

from pyarc.qcba.data_structures import QuantitativeDataFrame


df = pd.read_csv("./data/iris0.csv")

quant_dataframe = QuantitativeDataFrame(df)

ids = IDSOneVsAll()
ids.fit(quant_dataframe=quant_dataframe, debug=False)

acc = ids.score_auc(quant_dataframe)
```

optimizing for best lambda parameters using coordinate ascent, as described in the original paper

```python
import pandas as pd
from pyids.ids_classifier import IDS, mine_IDS_ruleset
from pyids.model_selection import CoordinateAscentOptimizer, train_test_split_pd

from pyarc.qcba.data_structures import QuantitativeDataFrame


df = pd.read_csv("./data/titanic.csv")
df_train, df_test = train_test_split_pd(df, prop=0.2)

ids_ruleset = mine_IDS_ruleset(df_train, rule_cutoff=50)

quant_dataframe_train = QuantitativeDataFrame(df_train)
quant_dataframe_test = QuantitativeDataFrame(df_test)

coordinate_ascent = CoordinateAscentOptimizer(IDS(), debug=True, maximum_delta_between_iterations=200, maximum_score_estimation_iterations=3)
coordinate_ascent.fit(ids_ruleset, quant_dataframe_train, quant_dataframe_test)

best_lambda_array = coordinate_ascent.current_best_params
```

or optimizing a One-vs-all IDS model

```python
import pandas as pd
from pyids.ids_classifier import IDSOneVsAll, mine_IDS_ruleset
from pyids.model_selection import CoordinateAscentOptimizer, train_test_split_pd

from pyarc.qcba.data_structures import QuantitativeDataFrame


df = pd.read_csv("./data/iris0.csv")
df_train, df_test = train_test_split_pd(df, prop=0.2)

ids_ruleset = mine_IDS_ruleset(df_train, rule_cutoff=50)

quant_dataframe_train = QuantitativeDataFrame(df_train)
quant_dataframe_test = QuantitativeDataFrame(df_test)

coordinate_ascent = CoordinateAscentOptimizer(IDSOneVsAll(), debug=True, maximum_delta_between_iterations=200, maximum_score_estimation_iterations=3)
coordinate_ascent.fit(ids_ruleset, quant_dataframe_train, quant_dataframe_test)

best_lambda_array = coordinate_ascent.current_best_params
```

using k-fold cross validation with AUC score

```python
import pandas as pd
from pyids.ids_classifier import IDSOneVsAll

dataframes = [ pd.read_csv("./data/iris{}.csv".format(i)) for i in range(10)]

kfold = KFoldCV(IDSOneVsAll(), dataframes, score_auc=True)
scores = kfold.fit(rule_cutoff=50)
```

using k-fold cross validation with accuracy score

```python
import pandas as pd
from pyids.ids_classifier import IDS

dataframes = [ pd.read_csv("./data/iris{}.csv".format(i)) for i in range(10)]

kfold = KFoldCV(IDS(), dataframes)
scores = kfold.fit(rule_cutoff=50)
```
