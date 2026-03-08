import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    out = []

    for el in x:
        if el > 0:
            out.append(el)
        else:
            out.append(alpha * (np.exp(el) - 1))

    return out