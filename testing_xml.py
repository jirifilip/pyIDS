import pandas as pd
from pyids import mine_IDS_ruleset

import xml.etree.ElementTree as ET

iris = pd.read_csv("./data/iris0.csv")

ids_ruleset = mine_IDS_ruleset(iris, 20)

ids_rules = list(ids_ruleset.ruleset)
tmp = ids_rules[0].to_dict()

rule0 = list(ids_ruleset.ruleset)[0]

ET.dump(rule0.to_ruleml_xml())

print(rule0.to_dict())