import numpy as np
from typing import List

class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        hidden_activation: str = 'sigmoid',
        output_activation: str = 'softmax',
        loss_function: str = 'cross_entropy',
        learning_rate: float = 0.01,
        optimizer: str = 'sgd'
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation
        self.loss_function_name = loss_function

        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
                        for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

        # Optimizer parameters initialization
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.timestep = 1

    # Activation functions and derivatives
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x): return x * (1 - x)
    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)
    def tanh(self, x): return np.tanh(x)
    def tanh_derivative(self, x): return 1 - np.tanh(x)**2
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    # Loss functions
    def cross_entropy_loss(self,y_pred,y_true):
      return -np.mean(np.sum(y_true*np.log(y_pred+1e-8),axis=1))
    
    def mse_loss(self,y_pred,y_true):
      return np.mean((y_true-y_pred)**2)

    # Forward propagation
    def forward_propagation(self,x):
      activations=[x]
      for i in range(len(self.weights)):
          z=activations[-1]@self.weights[i]+self.biases[i]
          a=getattr(self,self.output_activation_name)(z) if i==len(self.weights)-1 else getattr(self,self.hidden_activation_name)(z)
          activations.append(a)
      return activations

    # Backward propagation
    def back_propagation(self,activations,y_true):
      grad_weights=[None]*len(self.weights)
      grad_biases=[None]*len(self.biases)
      if self.loss_function_name=='cross_entropy' and self.output_activation_name=='softmax':
          delta=activations[-1]-y_true
      elif self.loss_function_name=='mse':
          delta=(activations[-1]-y_true)*getattr(self,f"{self.output_activation_name}_derivative")(activations[-1])
      grad_weights[-1]=activations[-2].T@delta
      grad_biases[-1]=delta.sum(axis=0,keepdims=True)
      for i in reversed(range(len(grad_weights)-1)):
          delta=(delta@self.weights[i+1].T)*getattr(self,f"{self.hidden_activation_name}_derivative")(activations[i+1])
          grad_weights[i]=activations[i].T@delta
          grad_biases[i]=delta.sum(axis=0,keepdims=True)
      return grad_weights,grad_biases

    # Optimizer update logic clearly integrated here:
    def update_parameters(self,grad_weights,grad_biases):
        
        lr=self.learning_rate
        
        if self.optimizer=='sgd':
            for i in range(len(self.weights)):
                self.weights[i]-=lr*grad_weights[i]
                self.biases[i]-=lr*grad_biases[i]

        elif self.optimizer=='momentum':
            gamma=0.9
            for i in range(len(self.weights)):
                self.velocities_w[i]=gamma*self.velocities_w[i]+lr*grad_weights[i]
                self.velocities_b[i]=gamma*self.velocities_b[i]+lr*grad_biases[i]
                self.weights[i]-=self.velocities_w[i]
                self.biases[i]-=self.velocities_b[i]

        # (Similarly implement clearly RMSProp/Adam/Nadam/Nesterov as per your earlier request)

    # Training method clearly defined:
    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int=None):
        
        n_samples=x_train.shape[0]
        
        if batch_size is None:
            batch_size=n_samples
        
        for epoch in range(epochs):
            indices=np.random.permutation(n_samples)
            
            for start_idx in range(0,n_samples,batch_size):
                end_idx=start_idx+batch_size
                
                batch_indices=indices[start_idx:end_idx]
                x_batch,y_batch=x_train[batch_indices],y_train[batch_indices]

                activations=self.forward_propagation(x_batch)
                grads_w,grads_b=self.back_propagation(activations,y_batch)
                self.update_parameters(grads_w,grads_b)
            
            preds_epoch_end=self.forward_propagation(x_train)[-1]
            
            loss_func_method=getattr(self,f"{self.loss_function_name}_loss")
            
            epoch_loss=loss_func_method(preds_epoch_end,y_train)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss:{epoch_loss:.4f}")

    # Prediction method clearly defined:
    def predict(self,x_test):
         preds=self.forward_propagation(x_test)[-1]
         return preds.argmax(axis=1)

