import numpy as np
from typing import List, Callable

class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        hidden_activation: str = 'sigmoid',
        output_activation: str = 'softmax',
        loss_function: str = 'cross_entropy',
        learning_rate: float = 0.1,
        optimizer: str = 'sgd',
        weight_initialization: str = 'random', # Added parameter
        weight_decay: float = 0.0
    ):
        """
        Initialize the neural network with given parameters.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation
        self.loss_function_name = loss_function
        self.weight_initialization = weight_initialization.lower()
        self.weight_decay = weight_decay
        # Validate activation and loss functions
        supported_activations = ['sigmoid', 'relu', 'tanh']
        supported_losses = ['cross_entropy', 'mse']
        supported_optimizers = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
        supported_initializations = ['random', 'xavier']

        if hidden_activation not in supported_activations:
            raise ValueError(f"Unsupported hidden activation '{hidden_activation}'. Supported: {supported_activations}")
        
        if output_activation not in ['softmax', 'sigmoid']:
            raise ValueError("Output activation must be either 'softmax' or 'sigmoid'.")

        if loss_function not in supported_losses:
            raise ValueError(f"Unsupported loss function '{loss_function}'. Supported: {supported_losses}")
        
        if optimizer not in supported_optimizers:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Supported: {supported_optimizers}")
        
        if self.weight_initialization not in supported_initializations:
            raise ValueError(f"Unsupported initialization '{weight_initialization}'. Supported: {supported_initializations}")


        if self.loss_function_name == 'mse':
            self.output_activation_name = 'sigmoid'

        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights, self.biases = self.initialize_weights(layer_sizes=layer_sizes, method='xavier')




    def initialize_weights(self, layer_sizes: List[int], method: str='random'):
        weights, biases = [], []
        
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]

            if method == 'random':
                w = np.random.randn(n_in, n_out) * 0.01
                weights.append(w)

            elif method == 'xavier':
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
                w_shape=(layer_sizes[i],layer_sizes[i+1])
                weights.append(np.random.uniform(-limit, limit, size=w_shape))

            else:
                raise ValueError("Unsupported initialization method. Choose from ['random', 'xavier'].")

            biases.append(np.zeros((1, layer_sizes[i+1])))

        return weights, biases
    

    # Activation functions and derivatives
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return self.stable_sigmoid(x)
        # return np.exp(x) / (1 + np.exp(x))
        # return 1 / (1 + np.exp(-x))
    
    def stable_sigmoid(self , x):
        positive_mask = x >= 0
        negative_mask = ~positive_mask
        result = np.empty_like(x)

        result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
        exp_x = np.exp(x[negative_mask])
        result[negative_mask] = exp_x = exp_x / (1 + exp_x)
        
        return result


    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    # def softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        # return x * (1 - x)

    # def softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        # s = x.reshape(-1, 1)
        # return np.diagflat(s) - np.dot(s, s.T)

    
    # Loss functions
    def cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def mse_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_true - y_pred)**2)

