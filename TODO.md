# TODO list

## Experiments

### 1. Mutual Information estimation
* [ ] Binning for binomial vectors
* [ ] Bound based MI estimation
* [ ] Improved bounds

### 2. Normalization techniques
* [ ] Different ordering of normalization
* [ ] Effect of normalization
* [ ] Other normalization techniques

### 3. BNN specific
* [ ] Effect of SGD and Bop optimizers
* [ ] Effect of different quantizers
  * [ ] SteSign
  * [ ] ApproxSign
  * [ ] SwishSign

### 4. Effect of input value range
* [ ] Input scaled in [0, 1]
* [ ] Input scaler in [-1,  +1]

## Coding
* [ ] Implement zookeeper experiment
* [ ] Integrate sacred, with file logging as default
* [ ] Optimizer state reload on loading experiments
* [ ] Sacred MongodB integration
* [ ] Custom callback to monitor progress?