from functions_toolkit import cross_entropy, one_hot
import numpy as np
from model import EdNet

# File contains different kind of optimizers for training a neural network from the model.py file. 

def GD_optimizer(model, train_features, train_targets, lr, max_iter = 10):
    # Gradient descent optimizer (full batch), as input the model from class EdNet is needed!
    train_targets = one_hot(train_targets, model.output_nodes)
    train_loss = []
    for iter in range(max_iter):
        print("Iteration -- ", iter)
        dW, db, z_out = model.backward_pass(train_targets, train_features)
        for idx in range(len(dW)):
            model.W[idx] = model.W[idx] - lr * dW[idx]
            model.b[idx] = model.b[idx] - lr * db[idx]
        train_loss.append(cross_entropy(z_out[model.hidden_layers + 1], train_targets))
    return model.W, model.b, train_loss


def iterate_minibatches(train_features, train_targets,num_classes, batchsize):
    # Minibatches creation function
    indeces = np.arange(train_features.shape[0])
    np.random.shuffle(indeces)
    y_one_hot = one_hot(train_targets,num_classes)
    for start_idx in range(0, train_features.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, train_features.shape[0])
        excerpt = indeces[start_idx:end_idx]
        yield train_features[excerpt], y_one_hot[excerpt]


def SGD_optimizer(model, train_features, train_targets, lr, batchsize, epochs = 100):
    # Stochastic Gradient Descent optimizer, as input the model from class EdNet is needed!
    train_loss = []
    for iter in range(epochs):
        print("Epochs--",iter)
        loss_epoch = 0
        for minibatch_X, minibatch_y in iterate_minibatches(train_features, train_targets, model.output_nodes, batchsize):
            dW, db, z_out = model.backward_pass(minibatch_y,minibatch_X)
            for idx in range(len(dW)):
                model.W[idx] = model.W[idx] - lr * dW[idx]
                model.b[idx] = model.b[idx] - lr * db[idx]
            loss_epoch += cross_entropy(z_out[model.hidden_layers + 1], minibatch_y)
        train_loss.append(loss_epoch/(train_features.shape[0]/batchsize))
    return model.W, model.b, train_loss