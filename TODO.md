# TODO list

## Experiments

### 1. Mutual Information estimation
* [ ] Binning for binomial vectors
* [ ] Bound based MI estimation
* [ ] Improved bounds
* [ ] MI estimation for train data also

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
* [x] Input scaled in [0, 1]
* [x] Input scaler in [-1,  +1]

## Coding
### 1. Train script
* [x] Implement zookeeper experiment
* [x] Integrate sacred, with file logging as default
* [ ] Optimizer state reload on loading experiments
* [ ] ~~Sacred MongodB integration~~
* [x] Custom callback to monitor progress?

### 2. Results visualization
* [ ] Notebook for viewing logged MI values
* [ ] Plotly for visualization