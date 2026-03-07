import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor=np.array(anchor)
    positive=np.array(positive)
    negative=np.array(negative)

    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)

    dap=np.sum((anchor-positive)**2,axis=1)
    dan=np.sum((anchor-negative)**2,axis=1)
    
    loss = np.maximum(0, dap - dan + margin)
    
    return float(np.mean(loss))