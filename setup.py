from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
  name = 'pyids',
  packages = ['pyids', "pyids.data_structures", "pyids.model_selection", "pyids.algorithms", "pyids.algorithms.optimizers", "pyids.utils", "pyids.rule_mining"],
  install_requires=['pandas', 'numpy', 'sklearn','pyarc', 'pyfim'],
  version = '0.0.1'
)
