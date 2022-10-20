import numpy as np

# loss function
def MSE(y, tx, w):
    # TODO  
    err = y - tx @ w
    return 1/2 * np.mean(err**2)


def compute_gradient(y, tx, w):
   # TODO 
    err = y - tx.dot(w)
    grad = -tx.T.dot(err)/len(err)
    return  grad, err