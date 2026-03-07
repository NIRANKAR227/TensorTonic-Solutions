import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern{captured by mask}).
    """
    # Write code here
    x=np.array(x)

    mask = (rng.random(x.shape) < (1-p)) / (1-p)  # /(1-p)scaled numeric mask
    output=x*mask

    return (output,mask)

'''
For Generator.random():

rng.random(x.shape)
    works because this function expects a tuple shape.
'''