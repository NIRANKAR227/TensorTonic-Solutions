import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    if a.ndim == 1:
        a = a.reshape(1, -1)     # For maintaing batch dimension
    if b.ndim == 1:
        b = b.reshape(1, -1)

        
    d=np.sqrt(np.sum(((a-b)**2),axis=1))
    loss=(y*(d**2))+((1-y)*np.maximum(0,margin-d)**2)

    if reduction=="sum":
        return np.sum(loss)
    else:
        return np.mean(loss)