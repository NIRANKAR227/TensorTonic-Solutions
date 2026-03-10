import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x=np.array(x)
    def sigmoid(x):
        x=np.array(x)
        return np.where(x>=0,(1/(1+np.exp(-x))),(np.exp(x)/(np.exp(x)+1)))
    
    return x*sigmoid(x)
    