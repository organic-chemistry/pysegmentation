pysegmentation
==============

Starting block of segmentation for simulated data using Unet

Description
===========

I started with unet because it is 'easier' to deal with the data.
For the Convolution layer there is a need to slide along the data.
When loading the data now all data below the read_size (once binned) are removed
In practice it could be padded for x with whatever value and I think
that I read in the cross_entropy help that if for y we assign the class -100 it is
discarded.

For the example I setted the bin_size to 200 , but it was a 'mistake' , I think we can
start with 100

Install
===========
```
conda create -n pysegmentation python=3.9 pytorch jupyterlab pytorch-lightning matplotlib pandas -c conda-forge
conda activate pysegmentation
pip install -e ./
```
or
```
conda create --name simunano --file environment.yml
```


Example
=========
There is a [notebook](notebook/Example.ipynb) in notebook to have an example for training

Train
=========
```
conda activate pysegmentation
python scripts/training.py  --data ~/simuNano/meg3_mock/learning_test --default_root_dir first_test
```
It will create first_test/lightning_logs/version_0/checkpoints/ folder were the training weights are located
