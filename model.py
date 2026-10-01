import numpy as np
from functions_toolkit import reLU, softmax, one_hot, reLU_derivative

# Class that creates a neural network. Input: number of hidden layers, nodes per layer (same for every layer), output nodes and input size

class EdNet:
    def __init__(self, hidden_layers, nodes_per_layer, output_nodes, input_size):
        self.hidden_layers = hidden_layers 
        self.nodes_per_layer = nodes_per_layer
        self.output_nodes = output_nodes
        self.input_size = input_size
        self.W, self.b = self.weight_init()

    # Function initializes the weights and biases of the neural network according to the Hu initialization.
    def weight_init(self):
        sigma = 2/self.input_size
        layer_sizes = ( [self.input_size] + [self.nodes_per_layer] * self.hidden_layers + [self.output_nodes]) # Format the layers with list operations
        W, b = {}, {}
        for idx in range(len(layer_sizes)-1):
            W[idx] = np.random.normal(0, sigma, 
                                             size = (layer_sizes[idx], layer_sizes[idx + 1]))
            b[idx] = np.random.normal(0, sigma,
                                             size = (1, layer_sizes[idx + 1]) )
        return W, b

    # Function that does the forward pass. Computation example for a neural network with the following configurations:
    # hidden_layers = 2, nodes_per_layer = 16, output_nodes = 10, input_size = 784
    # Compute phase: 1. z_in0: N x 16 + 1 x 16 (due to broadcasting this operation is valid: N x 16 + N x 16): N x 16, z_out1: reLU(z_in0) (z_out0: train_features)
    #                2. z_in1: N x 16 + 1 x 16 (broadcasting): N x 16, z_out2: reLU(z_in1)
    #                3. z_in2: N x 10 + 1 x 10 (broadcasting): N x 10, z_out3: reLU(z_in2)
    #                4. z_out3: Softmax probabilities, N x 10 -> one row is one observation and it is 1 x 10
    def forward_pass(self, train_features):
        z_in, z_out = {}, {} 
        z_out[0] = train_features.reshape(len(train_features),-1) # N (# train data) x input_size (so every observation is 1 x input_size) 
        for idx in range(self.hidden_layers):
            z_in[idx] = z_out[idx] @ self.W[idx] + self.b[idx]
            z_out[idx + 1] = reLU(z_in[idx])
        z_in[idx + 1] = z_out[idx + 1] @ self.W[idx + 1] + self.b[idx + 1]
        z_out[idx + 2] = softmax(z_in[idx + 1])
        return z_in, z_out

    # Function that computes that backpropagation gradients in a recursion manner. Computation example for a neural network with the following configurations:
    # hidden_layers = 2, nodes_per_layer = 16, output_nodes = 10, input_size = 784
    # Compute phase: 1. delta2: N x 10, dW2: 16 x N @ N 10 = 16 x 10, db2: (10 x N @ N x 1)^T = 1 x 10 (for the broadcasting) 
    #                2. delta1: N x 10 @ 10 x 16 * N x 16, dW1: 16 x N @ N x 16 = 16 x 16, db1: (16 x N @ N x 1)^T = 1 x 16 (for broadcasting)
    #                3. delta0: N x 16 @ 16 x 16 * N x 16, dW0: 784 x N @ N x 16  = 784 x 16, db0: (16 x N @ N x 1)^T = 1 x 16 (for broadcasting)
    def backward_pass(self, train_targets, train_features):
        dW, db, delta = {}, {}, {}
        z_in, z_out = self.forward_pass(train_features)
        ones_vec = np.ones((train_targets.shape[0],1))
        delta[self.hidden_layers] = z_out[self.hidden_layers+1] - train_targets
        dW[self.hidden_layers] = z_out[self.hidden_layers].T @ delta[self.hidden_layers] / train_targets.shape[0]
        db[self.hidden_layers] = (delta[self.hidden_layers].T @ ones_vec).T / train_targets.shape[0]
        for l in range(self.hidden_layers-1,-1,-1):
            delta[l] = delta[l+1] @ self.W[l+1].T * reLU_derivative(z_in[l])
            dW[l] = z_out[l].T @ delta[l]/ train_targets.shape[0]
            db[l] = (delta[l].T @ ones_vec).T / train_targets.shape[0]
        return dW, db, z_out