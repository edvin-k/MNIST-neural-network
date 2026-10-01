from feedforward_pass import forward_pass
from functions_toolkit import reLU_derivative
import numpy as np
def backprop(train_data,y_one_hot,W,B,V,D,U,C):
    X,z1in,z1out,z2in,z2out,fin,fout = forward_pass(train_data,W,B,V,D,U,C)
    delta_f = fout - y_one_hot  # 60,000 x 10
    dU = z2out.T.dot(delta_f)/X.shape[0] # 16 x 60,000 @ 60,000 10 = 16 x 10
    dC = delta_f.T.dot(np.ones((X.shape[0],1))).T/X.shape[0] # (10 x 60,000 @ 60,000 x 1)^T = 1 x 10 (for the broadcasting)
    delta2 = delta_f.dot(U.T) * reLU_derivative(z2in) # 60,000 x 10 @ 10 x 16 * 60,000 x 16
    dV = z1out.T.dot(delta2)/X.shape[0] # 16 x 60,000 @ 60,000 x 16 = 16 x 16 
    dD = delta2.T.dot(np.ones((X.shape[0],1))).T/X.shape[0] # (16 x 60,000 @ 60,000 x 1)^T = 1 x 16 (for broadcasting)
    delta1 = delta2.dot(V.T) * reLU_derivative(z1in) # 60,000 x 16 @ 16 x 16 * 60,000 x 16 
    dW = X.T.dot(delta1)/X.shape[0] # 784 x 60,000 @ 60,000 x 16  = 784 x 16
    dB = delta1.T.dot(np.ones((X.shape[0],1))).T/X.shape[0] # (16 x 60,000 @ 60,000 x 1)^T = 1 x 16 (for broadcasting)
    return dW,dB,dV,dD,dU,dC,fout