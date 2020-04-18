# Information Bottleneck in Binary World

## Setup environment
Following are the requirements. 
```
0. Python (v3.7.4)
1. tensorflow (v2.1)
2. larq (v0.9.3)
3. zookeeper (==v0.5.5)
4. sacred (v0.8.0)
5. matplotlib (v3.1.1)
```
In braces are the version used while testing. `==` denotes specific version to be used for reproduction. For other, there may be a option of using a higher version, but is not tested.

You can also use the conda environment file `conda-env.yml` provided to setup the system with tensorflow. 
```
conda create -f conda-env.yml
```
We used `tensorflow` from `pip` rather than conda as we observed it provided better speed in CPU system than the `tensorflow` package from `conda` channels.



## Run experiments
Basic run
```
TF_CPP_MIN_LOG_LEVEL=2 PYTHONPATH=./src/ python -m classify train fcn --dataset mnist --logdir ./zoo/
```

## Setting up MongoDB and Omniboard (optional)
