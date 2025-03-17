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

        Parameters:
        - input_size (int): Number of input features.
        - hidden_layers (List[int]): List containing the number of neurons in each hidden layer.
        - output_size (int): Number of output neurons.
        - hidden_activation (str): Activation function for hidden layers ('sigmoid', 'relu', 'tanh').
        - output_activation (str): Activation function for output layer ('softmax', 'sigmoid').
        - loss_function (str): Loss function ('cross_entropy', 'mse').
        - learning_rate (float): Learning rate for weight updates.
        - optimizer (str): Optimization algorithm ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam').
        - weight_initialization (str): Weight initialization method ('random', 'xavier').
        - weight_decay (float): Regularization term to prevent overfitting.
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

        


        # Optimizer parameters initialization
        self.momentum_gamma = 0.9
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]
        
        # For Adam/Nadam/RMSProp
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        
        # Timestep for Adam/Nadam
        self.timestep = 1


###################################################################################################################
    # Weight initialization


    def initialize_weights(self, layer_sizes: List[int], method: str='random'):
        """
        Initializes weights and biases using the specified method.
        
        Parameters:
        - layer_sizes (List[int]): List containing the number of neurons in each layer.
        - method (str): Weight initialization method ('random' or 'xavier').
        
        Returns:
        - weights (List[np.ndarray]): List of weight matrices for each layer.
        - biases (List[np.ndarray]): List of bias vectors for each layer.
        """
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

###################################################################################################################

    # Activation functions and derivatives #

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid activation function.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the sigmoid function.
        """

        return self.stable_sigmoid(x)
        # return np.exp(x) / (1 + np.exp(x))
        # return 1 / (1 + np.exp(-x))
    
    def stable_sigmoid(self , x):
        """
        Numerically stable sigmoid function.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the sigmoid function.
        """

        positive_mask = x >= 0
        negative_mask = ~positive_mask
        result = np.empty_like(x)

        result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
        exp_x = np.exp(x[negative_mask])
        result[negative_mask] = exp_x = exp_x / (1 + exp_x)
        
        return result


    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the sigmoid function.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the derivative of sigmoid.
        """
        return x * (1 - x)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the ReLU activation function.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the ReLU function.
        """
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the ReLU function.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the derivative of ReLU.
        """
        return (x > 0).astype(float)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the tanh activation function.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the tanh function.
        """
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the tanh function.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the derivative of tanh.
        """
        return 1 - np.tanh(x)**2

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the stable softmax activation function to prevent overflow.
        
        Parameters:
        - x (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Output after applying the softmax function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    # def softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        # return x * (1 - x)

    # def softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        # s = x.reshape(-1, 1)
        # return np.diagflat(s) - np.dot(s, s.T)



###################################################################################################################
    
    # Loss functions

    def cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the cross-entropy loss between predicted and true labels.
        
        Parameters:
        - y_pred (np.ndarray): Predicted probabilities.
        - y_true (np.ndarray): True labels (one-hot encoded).
        
        Returns:
        - float: Computed cross-entropy loss.
        """
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def mse_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) loss.
        
        Parameters:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.
        
        Returns:
        float: Computed MSE loss.
        """
        return np.mean((y_true - y_pred)**2)
    
###################################################################################################################

    # Forward propagation

    def forward_propagation(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Perform forward propagation through the network.
        
        Parameters:
        x (np.ndarray): Input data.
        
        Returns:
        List[np.ndarray]: List of activations for each layer, including input.
        """
        activations = [x]
        
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            
            if i == len(self.weights) - 1:
                a = getattr(self, self.output_activation_name)(z)
            else:
                a = getattr(self, self.hidden_activation_name)(z)
            
            activations.append(a)
        
        return activations
    

###################################################################################################################    

    # Backward propagation

    def back_propagation(self, activations: List[np.ndarray], y_true: np.ndarray):
        """
        Perform backpropagation to compute gradients.
        
        Parameters:
        activations (List[np.ndarray]): List of activations from forward propagation.
        y_true (np.ndarray): True output values.
        
        Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Gradients for weights and biases.
        """
        
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        # Compute delta at output layer
        if self.loss_function_name == 'cross_entropy' and self.output_activation_name == 'softmax':
            delta = activations[-1] - y_true
        elif self.loss_function_name == 'mse':
            delta = (activations[-1] - y_true) * (activations[-1] * (1 - activations[-1]))
        
        grad_weights[-1] = activations[-2].T @ delta
        grad_biases[-1] = delta.sum(axis=0, keepdims=True)

        # Propagate backwards through hidden layers
        for i in reversed(range(len(grad_weights)-1)):
            delta = (delta @ self.weights[i+1].T) * getattr(self, f"{self.hidden_activation_name}_derivative")(activations[i+1])
            grad_weights[i] = activations[i].T @ delta
            grad_biases[i] = delta.sum(axis=0, keepdims=True)

        return grad_weights, grad_biases
    

###################################################################################################################

    # Optimizers

    def sgd(self,grad_weights,grad_biases):
        """
        Performs Stochastic Gradient Descent (SGD) optimization.
        """
        lr=self.learning_rate
        for i in range(len(self.weights)):
            #   self.weights[i]-=lr*grad_weights[i]
            #   self.biases[i]-=lr*grad_biases[i]


              self.weights[i] -= lr * (grad_weights[i] + self.weight_decay * self.weights[i])
              self.biases[i] -= lr * grad_biases[i]

    def momentum(self,grad_weights,grad_biases):
        """
        Performs Momentum-based optimization.
        """
        lr=self.learning_rate
        for i in range(len(self.weights)):
              self.velocities_w[i]=self.momentum_gamma*self.velocities_w[i]+lr*grad_weights[i]
              self.velocities_b[i]=self.momentum_gamma*self.velocities_b[i]+lr*grad_biases[i]

              self.weights[i]-=self.velocities_w[i]
              self.biases[i]-=self.velocities_b[i]

              if self.weight_decay>0:
                self.weights[i]-=lr*self.weight_decay*self.weights[i]

    def nesterov(self,grad_weights,grad_biases):
        """
        Performs nesterov-based optimization.
        """
        lr=self.learning_rate
        for i in range(len(self.weights)):
              self.v_weights[i]=self.beta2*self.v_weights[i]+(1-self.beta2)*(grad_weights[i]**2)
              self.v_biases[i]=self.beta2*self.v_biases[i]+(1-self.beta2)*(grad_biases[i]**2)

              w_update=lr*grad_weights[i]/(np.sqrt(self.v_weights[i])+self.epsilon)
              b_update=lr*grad_biases[i]/(np.sqrt(self.v_biases[i])+self.epsilon)

              self.weights[i]-=w_update
              self.biases[i]-=b_update

              if self.weight_decay>0:
                  self.weights[i]-=lr*self.weight_decay*self.weights[i]

    def rmsprop(self,grad_weights,grad_biases):
        """
        Performs RMSProp optimization.
        """
        lr=self.learning_rate
        for i in range(len(self.weights)):
              self.v_weights[i]=self.beta2*self.v_weights[i]+(1-self.beta2)*(grad_weights[i]**2)
              self.v_biases[i]=self.beta2*self.v_biases[i]+(1-self.beta2)*(grad_biases[i]**2)

              w_update=lr*grad_weights[i]/(np.sqrt(self.v_weights[i])+self.epsilon)
              b_update=lr*grad_biases[i]/(np.sqrt(self.v_biases[i])+self.epsilon)

              self.weights[i]-=w_update
              self.biases[i]-=b_update

              if self.weight_decay>0:
                  self.weights[i]-=lr*self.weight_decay*self.weights[i]

    def adam(self,grad_weights,grad_biases):
        """
        Performs ADAM optimization.
        """
        lr = self.learning_rate
        
        b1,b2,e=self.beta1,self.beta2,self.epsilon
        t=self.timestep;self.timestep+=1
        for i in range(len(self.weights)):
            m,v=self.m_weights,self.v_weights
            mb,vb=self.m_biases,self.v_biases
            gw,gb=grad_weights,grad_biases

            # gw[i]+=self.weight_decay*self.weights[i]

            m[i]=b1*m[i]+(1-b1)*gw[i]
            v[i]=b2*v[i]+(1-b2)*(gw[i]**2)

            mb[i]=b1*mb[i]+(1-b1)*gb[i]
            vb[i]=b2*vb[i]+(1-b2)*(gb[i]**2)

            mhat=m[i]/(1-b1**t)
            vhat=v[i]/(1-b2**t)

            mbhat=mb[i]/(1-b1**t)
            vbhat=vb[i]/(1-b2**t)

            w_update=lr*mhat/(np.sqrt(vhat)+e)
            b_update=lr*mbhat/(np.sqrt(vbhat)+e)

            self.weights[i]-=w_update
            self.biases[i]-=b_update

            if self.weight_decay > 0:
                self.weights[i] -= lr * self.weight_decay * self.weights[i]

    def nadam(self,grad_weights,grad_biases):
        """
        Performs Nadam optimization.
        """
        lr = self.learning_rate
        b1,b2,e=self.beta1,self.beta2,self.epsilon
        t=self.timestep;self.timestep+=1
        for i in range(len(self.weights)):
            m,v=self.m_weights,self.v_weights
            mb,vb=self.m_biases,self.v_biases
            gw,gb=grad_weights,grad_biases

            # gw[i]+=self.weight_decay*self.weights[i]

            m[i]=b1*m[i]+(1-b1)*gw[i]
            v[i]=b2*v[i]+(1-b2)*(gw[i]**2)

            mb[i]=b1*mb[i]+(1-b1)*gb[i]
            vb[i]=b2*vb[i]+(1-b2)*(gb[i]**2)

            mhat=m[i]/(1-b1**t)
            vhat=v[i]/(1-b2**t)

            mbhat=mb[i]/(1-b1**t)
            vbhat=vb[i]/(1-b2**t)

            mhat=b1*mhat+(1-b1)*gw[i]/(1-b1**t)
            mbhat=b1*mbhat+(1-b1)*gb[i]/(1-b1**t)

            w_update=lr*mhat/(np.sqrt(vhat)+e)
            b_update=lr*mbhat/(np.sqrt(vbhat)+e)

            self.weights[i]-=w_update
            self.biases[i]-=b_update

            if self.weight_decay > 0:
                self.weights[i] -= lr * self.weight_decay * self.weights[i]





###################################################################################################################

    # Parameter update using optimizers

    def update_parameters(self,grad_weights,grad_biases):
      """
        Update network parameters using the selected optimizer.
        
        Parameters:
        grad_weights (List[np.ndarray]): Gradients of weights.
        grad_biases (List[np.ndarray]): Gradients of biases.
        """
      lr=self.learning_rate
      
      if self.optimizer=='sgd':
          self.sgd(grad_weights,grad_biases)
       

      elif self.optimizer=='momentum':
          self.momentum(grad_weights,grad_biases)


      elif self.optimizer=='nesterov':
          self.nesterov(grad_weights,grad_biases)
       

      elif self.optimizer=='rmsprop':
          self.rmsprop(grad_weights,grad_biases)
       

      elif self.optimizer=='adam':
            self.adam(grad_weights,grad_biases)

      elif self.optimizer=='nadam':
            self.nadam(grad_weights,grad_biases)    
        
###################################################################################################################

    # Accuracy calculation

    def calculate_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate classification accuracy.
        
        Parameters:
        y_pred (np.ndarray): Predicted output values.
        y_true (np.ndarray): True labels.
        
        Returns:
        float: Accuracy percentage.
        """
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(pred_labels == true_labels) * 100
        return accuracy

###################################################################################################################

    # Training method
        
    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int=None,
            x_val: np.ndarray = None,
            y_val: np.ndarray = None,
            wandb=None):
        
        """
        Train the neural network.
        
        Parameters:
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training labels.
        epochs (int): Number of epochs to train.
        batch_size (int, optional): Size of training batches. Defaults to full dataset.
        x_val (np.ndarray, optional): Validation input data.
        y_val (np.ndarray, optional): Validation labels.
        wandb: Weights and Biases logging.
        """
        
        n_samples = x_train.shape[0]
        
        if batch_size is None:
            batch_size = n_samples
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                
                batch_indices = indices[start_idx:end_idx]
                x_batch, y_batch = x_train[batch_indices], y_train[batch_indices]

                activations = self.forward_propagation(x_batch)
                grads_w, grads_b = self.back_propagation(activations, y_batch)
                self.update_parameters(grads_w, grads_b)
            
            preds_epoch_end = self.forward_propagation(x_train)[-1]
            self.lit = preds_epoch_end
            loss_func_method : Callable[[np.ndarray,np.ndarray],float]= getattr(self,f"{self.loss_function_name}_loss")
            
            # epoch_loss=loss_func_method(preds_epoch_end,y_train)
            epoch_loss = loss_func_method(preds_epoch_end, y_train)

            # Calculate accuracy
            accuracy = self.calculate_accuracy(preds_epoch_end, y_train)
            
            # Print both loss and accuracy
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

            if x_val is not None and y_val is not None:
                val_preds = self.forward_propagation(x_val)[-1]
                val_loss_func = getattr(self, f"{self.loss_function_name}_loss")
                val_loss = val_loss_func(val_preds, y_val)
                val_acc = self.calculate_accuracy(val_preds, y_val)

                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
                if wandb:
                    wandb.log({"Train_Loss": epoch_loss, "Train_Accuracy": accuracy, "Val Loss": val_loss, "Val Accuracy": val_acc})

            # print(f"Epoch {epoch+1}/{epochs}, Loss:{epoch_loss:.4f}")


###################################################################################################################

    # Prediction method
    
    def predict(self,x_test:np.ndarray)->np.ndarray:
         """
        Make predictions on test data.
        
        Parameters:
        x_test (np.ndarray): Test input data.
        
        Returns:
        np.ndarray: Predicted class labels.
        """
        
         preds=self.forward_propagation(x_test)[-1]
         return preds.argmax(axis=1)

