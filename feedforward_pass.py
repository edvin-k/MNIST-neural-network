def forward_pass(train_data,W,B,V,D,U,C):
    X = train_data.reshape(len(train_data),-1) # 60,000 (# train data) x 784 (so every observation is 1 x 784)
    z1in = X.dot(W) + B # 60,000 x 16 + 1 x 16 (due to broadcasting this operation is valid: 60,000 x 16 + 60,000 x 16): 60,000 x 16 
    z1out = reLU(z1in) 
    z2in = z1out.dot(V) + D # 60,000 x 16 + 1 x 16 (broadcasting): 60,000 x 16
    z2out = reLU(z2in) 
    fin = z2out.dot(U) + C # 60,000 x 10 + 1 x 10 (broadcasting): 60,000 x 10
    fout = softmax(fin) # 60,000 x 10 -> one row is one observation and it is 1 x 10, the softmax probabilities
    return X,z1in,z1out,z2in,z2out,fin,fout