==============
pysegmentation
==============


Add a short description here!


Description
===========

A longer description of your project goes here...


.. _pyscaffold-notes:

Install
===========
conda create -n pysegmentation python=3.9 pytorch jupyterlab pytorch-lightning matplotlib pandas -c conda-forge
conda activate pysegmentation
pip install -e ./
pip install setuptools==59.5.0

Example
=========
There is a [notebook](notebook/Example.ipynb) in notebook to have an example for training 

Train
=========
```
python scripts/training.py  --data ~/simuNano/meg3_mock/learning_test --default_root_dir first_test
```
It will create first_test/lightning_logs/version_0/checkpoints/ folder were the training weights are located
