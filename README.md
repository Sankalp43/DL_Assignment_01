# DA6400 Assignment 01

## Report Link (WANDB)
[View Report](https://wandb.ai/da24s021-indian-institute-of-technology-madras/DA6401_assignment1/reports/DA6401-Assignment-1--VmlldzoxMTcxMzEwNw)

## Main Files for Assignment Grading
The assignment consists of three main files:

- **`train.py`** - Accepts command-line arguments for training.
- **`neural_network.py`** - Contains the implementation of the neural network.
- **`train.ipynb`** - Includes solutions for assignment questions, utilizing `neural_network.py` for computations.

The code is well-documented with inline comments and structured in a modular way, making it easy to understand and extend. The neural network implementation resides in `neural_network.py` as a single class containing all necessary methods for training and evaluation. The `train.py` script serves as a wrapper, leveraging the `NeuralNetwork` class to train models and upload reports to Weights & Biases (WANDB).

### Key Features
- Modular design for easy integration of new features such as activation functions, loss functions, and optimizers.
- Comprehensive documentation and inline comments for better readability.
- WANDB integration for experiment tracking.

## Neural Network Initialization
To initialize the neural network:

```python
from neural_network import NeuralNetwork

# Initialize the neural network
nn = NeuralNetwork(
    input_size=28 * 28,
    hidden_layers=[256, 128, 64],
    output_size=10,
    weight_initialization="xavier",
    hidden_activation='sigmoid',
    learning_rate=0.001,
    optimizer='nadam',
    loss_function='cross_entropy',
    weight_decay=0
)
```

### Forward Propagation (Predictions)
```python
pred = nn.predict(test_images)
```

### Training the Neural Network
```python
nn.train(
    train_images,
    train_labels,
    epochs=10,
    batch_size=32,
    x_val=test_images,
    y_val=test_labels,
    wandb=run
)
```

## Running the Training Script
You can execute `train.py` to train the neural network using the following command:

> **Note:** The trained model is not saved automatically.

```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```

### Command-Line Arguments for `train.py`

```python
('-wp', '--wandb_project', type=str, default='DA6401_assignment1', help='Project name for WANDB tracking')
('-we', '--wandb_entity', type=str, default='myname', help='WANDB entity for experiment tracking')
('-sid', '--wandb_sweepid', type=str, default=None, help='WANDB Sweep ID for logging sweep runs')
('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset selection')
('-e', '--epochs', type=int, default=10, help='Number of training epochs')
('-b', '--batch_size', type=int, default=64, help='Batch size for training')
('-l', '--loss', type=str, default='cross_entropy', choices=['mse', 'cross_entropy'], help='Loss function')
('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'], help='Optimizer selection')
('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate')
('-m', '--momentum', type=float, default=0.9, help='Momentum for applicable optimizers')
('-beta', '--beta', type=float, default=0.99, help='Beta for RMSProp optimizer')
('-beta1', '--beta1', type=float, default=0.9, help='Beta1 for Adam and Nadam optimizers')
('-beta2', '--beta2', type=float, default=0.999, help='Beta2 for Adam and Nadam optimizers')
('-eps', '--epsilon', type=float, default=1e-08, help='Epsilon value for optimizers')
('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay for regularization')
('-w_i', '--weight_init', type=str, default='random', choices=['random', 'xavier'], help='Weight initialization method')
('-nhl', '--num_layers', type=int, default=5, help='Number of hidden layers')
('-sz', '--hidden_size', type=int, default=128, help='Number of neurons in hidden layers')
('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'], help='Activation function')
```

## Submitted By:
**Sankalp Shrivastava**
**DA24S021**

