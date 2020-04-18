import numpy as np

def compute_mi(label_idx, layer_act, binsize):
    """
        Compute mutual information based on binning.
        Based on : MI computation code from https://github.com/ravidziv/IDNNs
        Arguments:
            label_idx   : A dictionary of format 
                { 
                    label: numpy array of boolean values corresponding to indices,
                    ...
                }
            layer_act   : layer activation data for each test sample. [n_samples x n_act].
            binsize     : size of the bin (scalar)
        Returns:
            H(layer) to be used as proxy for I(T;X) and I(T;Y)
    """
    # compute entropy of activations -- proxy for I(T;X)
    H_T = compute_entropy(layer_act, 0.5)
    
    # compute entropy of activation given label
    H_TY = 0
    for label, idx in label_idx.items():
        H_TY += idx.mean() * compute_entropy(layer_act[idx, :], 0.5)

    # return I(T;X), I(T;Y)
    return H_T, H_T - H_TY


def compute_entropy(data, binsize):
    # digitize data to form discrete values
    data_discrete = np.floor(data / binsize).astype('int')
    # compute empirical probability
    p_est = get_unique_probs(data_discrete)
    # return entropy
    return -np.sum(p_est * np.log2(p_est))


def get_unique_probs(x):
    # create unique ids for each values
    unique_ids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    # count 
    _, unique_counts = np.unique(unique_ids, return_counts=True, return_index=False, return_inverse=False)
    # empirical probability
    return unique_counts / unique_counts.sum()


# tests
if __name__ == "__main__":

    def print_stats(d, binsize=0.25):
        # compute empirical density
        p_est = get_unique_probs(np.floor(d / binsize).astype('int'))
        print("p_est::", p_est.shape, "=", p_est)
        # compute empirical entropy
        h = compute_entropy(d, binsize)
        print("entropy =", h)


    # test 1
    d = np.random.random((10000, 2))
    print("with bin_size = 0.25")
    print_stats(d, 0.25)
    print("with bin_size = 0.10")
    print_stats(d, 0.10)
    
    # test 2
    d = np.random.randn(10000, 2)
    print("with bin_size = 0.25")
    print_stats(d, 0.25)
    print("with bin_size = 0.10")
    print_stats(d, 0.10)

    