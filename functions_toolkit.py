def reLU(x):
    return np.maximum(0,x)

def reLU_derivative(x):
    r = np.ones(x.shape)
    r[x <= 0] = 0
    return r


def softmax(x):
    c = np.max(x,axis=1).reshape(x.shape[0],1)
    num = np.exp(x - c)
    denom = num.sum(axis=1).reshape(x.shape[0],1)
    return num/denom
    

def one_hot(y):
    num_categories = len(np.unique(y))
    y_one_hot = np.zeros((y.size,num_categories))
    y_one_hot[np.arange(y.size),y] = 1
    return y_one_hot

def cross_entropy(fout, train_data, y_one_hot): 
    loss =  (-y_one_hot * np.log(fout)).sum(axis = 1) # (60,000 x 10 * 60,000 x 10).sum(..) = 60,000 x 1
    avg_risk = loss.sum(axis = 0)/train_data.shape[0] # 1x1 , a summed (avg) loss over all observations 
    return avg_risk