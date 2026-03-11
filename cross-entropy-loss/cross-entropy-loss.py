import numpy as np
from sklearn.preprocessing import OneHotEncoder

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here

    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

    prob=y_pred[np.arange(len(y_true)),y_true]  # y_pred[row,column]

    loss=np.log(prob)

    return np.mean(-loss)










    
    '''encoded=OneHotEncoder(sparse_output=False)
    encoded_y_true=encoded.fit_transform(np.array(y_true).reshape(-1,1))
    
    y_pred=np.array(y_pred)
    
    cel=np.sum(encoded_y_true*np.log(y_pred),axis=1)
    return np.mean(-cel)'''