import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import copy
from optimizers import GD_optimizer, SGD_optimizer
from model import EdNet


(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Target hidden layer counts
hidden_layers_list = [2, 10, 20]

# Optimizer configurations
experiments = {
    "GD": (GD_optimizer, dict(lr=0.001, max_iter=20)),
    "SGD (batch 64)": (SGD_optimizer, dict(lr=0.001, batchsize=64, epochs=20)),
}

# Create 1 row x 3 columns of subplots with a shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for i, n_layers in enumerate(hidden_layers_list):
    ax = axes[i]
    
    # Build base model for this depth so all optimizers start from identical weights
    base_model = EdNet(n_layers, 16, 10, 784)

    for name, (optimizer, kwargs) in experiments.items():
        print(f"Training [{n_layers} layers] - {name}")
        net = copy.deepcopy(base_model)
        
        _, _, losses = optimizer(net, train_X, train_y, **kwargs)

        _, z_out = net.forward_pass(test_X)
        predicted = np.argmax(z_out[net.hidden_layers + 1], axis=1)
        accuracy = np.mean(predicted == test_y)

        print(f"   final train loss: {losses[-1]:.4f} | test accuracy: {accuracy:.4f}")

        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, label=f"{name} (acc: {accuracy:.3f})")

    ax.set_title(f"{n_layers} Hidden Layers")
    ax.set_xlabel("Epochs")
    ax.grid(True)
    ax.legend()

# Set common y-axis label on the leftmost plot
axes[0].set_ylabel("Train Loss")

plt.suptitle("Training Loss Across Network Depths", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()