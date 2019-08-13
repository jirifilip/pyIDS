import pandas as pd
from pyids.ids_classifier import mine_IDS_ruleset

import xml.etree.ElementTree as ET

iris = pd.read_csv("./data/iris0.csv")

ids_ruleset = mine_IDS_ruleset(iris, 20)

ids_rules = list(ids_ruleset.ruleset)
tmp = ids_rules[0].to_dict()

et = ids_ruleset.to_xml()

et.write("tmp.xml")
