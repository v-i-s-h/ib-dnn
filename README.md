# Information Bottleneck in Binary World

## Setup environment
Following are the requirements
```
0. Python (v3.7)
1. tensorflow (v2.1)
2. larq
3. zookeeper (v0.5.5)
4. sacred
```

You can also use the conda environment file `conda-env.yml` provided to setup the system with tensorflow. 
```
conda create -f conda-env.yml
```
We used `tensorflow` from `pip` rather than conda as we observed it provided better speed in CPU system than the `tensorflow` package from `conda` channels.



## Run experiments
