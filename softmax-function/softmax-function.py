import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x=np.array(x)
    # adjusted Xi (Xi-max(x))

    adjusted_x=x-np.max(x,axis=-1,keepdims=True)

    softmax_out= (np.exp(adjusted_x)/np.sum(np.exp(adjusted_x),axis=-1,keepdims=True))

    return softmax_out