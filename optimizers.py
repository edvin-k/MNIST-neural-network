sigma = 2/784 # Hu initialization 
W_init = np.random.normal(0,sigma , size=(784,16)) # 784 x 16
B_init = np.random.normal(0,sigma , size=(1,16)) # 1 x 16 
V_init = np.random.normal(0,sigma , size=(16,16)) # 16 x 16
D_init = np.random.normal(0,sigma , size=(1,16)) # 1 x 16
U_init = np.random.normal(0,sigma , size=(16,10)) # 16 x 10 
C_init = np.random.normal(0,sigma , size=(1,10)) # 1 x 10 

def gradient_descent_optimizer(train_data, target, lr, max_iter = 5000):
    W,B,V,D,U,C = W_init,B_init,V_init,D_init,U_init,C_init
    y_one_hot = one_hot(target)
    avg_risk = []
    for iter in range(max_iter):
        print("Iteration--",iter)
        dW,dB,dV,dD,dU,dC,fout = backprop(train_data, y_one_hot,W,B,V,D,U,C)
        avg_risk.append(cross_entropy(fout,train_data, y_one_hot))
        U = U - lr*dU
        C = C - lr*dC
        V = V - lr*dV
        D = D - lr*dD
        W = W - lr*dW
        B = B - lr*dB   
    return W,B,V,D,U,C,avg_risk



def iterate_minibatches(train_data, target, batchsize):
    indeces = np.arange(train_data.shape[0])
    np.random.shuffle(indeces)
    y_one_hot = one_hot(target)
    for start_idx in range(0, train_data.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, train_data.shape[0])
        excerpt = indeces[start_idx:end_idx]
        yield train_data[excerpt], y_one_hot[excerpt]


def SGD_optimizer(train_data, target, lr, batchsize, epochs = 5000):

    W,B,V,D,U,C = W_init,B_init,V_init,D_init,U_init,C_init
    avg_risk = []
    for iter in range(epochs):
        print("Epochs--",iter)
        for minibatch_X, minibatch_y in iterate_minibatches(train_data, target, batchsize):
            dW,dB,dV,dD,dU,dC,fout = backprop(minibatch_X,minibatch_y,W,B,V,D,U,C)
            avg_risk.append(cross_entropy(fout,minibatch_X, minibatch_y))
            U = U - lr*dU
            C = C - lr*dC
            V = V - lr*dV
            D = D - lr*dD
            W = W - lr*dW
            B = B - lr*dB   
    return W,B,V,D,U,C,avg_risk