import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from keras.datasets import fashion_mnist , mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse
import wandb
import os




parser = argparse.ArgumentParser(description='Neural Network Training Configuration')



parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_assignment1', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-sid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset choices: ["mnist", "fashion_mnist"]')
parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used to train neural network.')
parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=["mse", "cross_entropy"], help='Loss function choices: ["mean_squared_error", "cross_entropy"]')
parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"], help='Optimizer choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum used by momentum and nag optimizers.')
parser.add_argument('-beta', '--beta', type=float, default=0.99, help='Beta used by rmsprop optimizer')
parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers.')
parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers.')
parser.add_argument('-eps', '--epsilon', type=float, default=1e-08, help='Epsilon used by optimizers.')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay used by optimizers.')
parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=["random", "xavier"], help='Weight initialization choices: ["random", "xavier"]')
parser.add_argument('-nhl', '--num_layers', type=int, default=5, help='Number of hidden layers used in feedforward neural network.')
parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a feedforward layer.')
parser.add_argument('-a', '--activation', type=str, default='sigmoid', choices=[ "sigmoid", "tanh", "relu"], help='Activation function choices: ["identity", "sigmoid", "tanh", "ReLU"]')


args = parser.parse_args()

run = wandb.init(
      project=args.wandb_project,
      )

wandb.run.name = f'lr_{args.learning_rate}_bs_{args.batch_size}_opt_{args.optimizer}_act_{args.activation}_loss_{args.loss}_wd_{args.weight_decay}_wi_{args.weight_init}_hl_{args.num_layers}_sz_{args.hidden_size}_wandbid_{wandb.run.id}'

# Load the Dataset
if args.dataset == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
else:
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Flatten the images
train_images = train_images.reshape((-1, 28 * 28))
test_images = test_images.reshape((-1, 28 * 28))


# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1 , random_state=42)




# Train the neural network
nn = NeuralNetwork(
        input_size= 28 * 28,
        hidden_layers = args.num_layers * [args.hidden_size],
        output_size = 10,
        hidden_activation = args.activation,
        output_activation = 'softmax',
        loss_function= args.loss,
        learning_rate= args.learning_rate,
        optimizer= args.optimizer,
        weight_initialization= args.weight_init,
        weight_decay= args.weight_decay
    )

    
    
    # 28 * 28,, 10,weight_initialization="xavier" ,hidden_activation='relu' ,  learning_rate=0.001, optimizer='sgd', loss_function='cross_entropy',weight_decay=0.001)
nn.train(
            x_train= train_images,
            y_train = train_labels,
            epochs = args.epochs,
            batch_size= args.batch_size,
            x_val= val_images,
            y_val= val_labels,
            wandb=run
            )
    
    





