# You are not allowed to import any other libraries or modules.

import torch
import torch.nn as nn

""" Fully Connected Layer """
    
class FCLayer(nn.Module):
    def __init__(self, num_input, num_output):
        """
        Initialize the Fully Connected (Linear) Layer.
      
        Args:
            num_input: Number of input features.
            num_output: Number of output features.
        """
        super(FCLayer, self).__init__()
        self.num_input = num_input
        self.num_output = num_output

        # Xavier initialization for weights
        self.W = nn.Parameter(torch.randn(num_input, num_output) * (2 / (num_input + num_output))**0.5)
        self.b = nn.Parameter(torch.zeros(1, num_output))
        self.X = None  # Store input for backward pass

    def forward(self, X):
        """
        Perform the forward pass.
        
        Args:
            X: Tensor of shape (batch_size, num_input), the input features.
        Returns:
            Tensor of shape (batch_size, num_output), the output after applying the linear transformation.
        """
        # Store the input for use in the backward pass
        self.X = X
        # Linear transformation
        return X @ self.W + self.b

    def backward(self, delta):
        """
        Perform the backward pass.
        
        Args:
            delta: Tensor of shape (batch_size, num_output), the gradient from the next layer.
        Returns:
            delta_next: Tensor of shape (batch_size, num_input), the gradient to pass to the previous layer.
        """
        # Gradient with respect to weights and biases
        self.dW = self.X.T @ delta
        self.db = delta.sum(dim=0, keepdim=True)
        
        # Gradient with respect to input (for previous layer)
        delta_next = delta @ self.W.T  # (batch_size, num_output) * (num_output, num_input) -> (batch_size, num_input)
        
        return delta_next



""" Sigmoid Layer """

class SigmoidLayer(nn.Module):
    def __init__(self):
        """
        Initialize the Sigmoid activation layer.
        """
        super(SigmoidLayer, self).__init__()
        self.Z = None  # Store output of the sigmoid for backward pass

    def forward(self, X):
        """
        Perform the forward pass using the Sigmoid function.
        
        Args:
            X: Tensor of shape (batch_size, num_features), the input features.
        Returns:
            Tensor of shape (batch_size, num_features), the output after applying the Sigmoid function.
        """
        # Apply the sigmoid function and store the output
        self.Z = 1 / (1 + torch.exp(-X))
        return self.Z

    def backward(self, delta):
        """
        Perform the backward pass.
        
        Args:
            delta: Tensor of shape (batch_size, num_features), the gradient from the next layer.
        Returns:
            delta_next: Tensor of shape (batch_size, num_features), the gradient to pass to the previous layer.
        """
        # Compute the gradient of the sigmoid function
        sigmoid_derivative = self.Z * (1 - self.Z)
        
        # Compute delta_next by element-wise multiplying the incoming delta with the sigmoid derivative
        delta_next = delta * sigmoid_derivative
        return delta_next



""" ReLU Layer """

import torch
import torch.nn as nn

class ReLULayer(nn.Module):
    def __init__(self):
        """
        Initialize the ReLU activation layer.
        """
        super(ReLULayer, self).__init__()
        self.X = None  # Store input for backward pass

    def forward(self, X):
        """
        Perform the forward pass using the ReLU function.
        
        Args:
            X: Tensor of shape (batch_size, num_features), the input features.
        Returns:
            Tensor of shape (batch_size, num_features), the output after applying ReLU (max(0, x)).
        """
        # Apply ReLU and store the input for use in the backward pass
        self.X = X
        return torch.maximum(torch.tensor(0.0), X)

    def backward(self, delta):
        """
        Perform the backward pass.
        
        Args:
            delta: Tensor of shape (batch_size, num_features), the gradient from the next layer.
        Returns:
            delta_next: Tensor of shape (batch_size, num_features), the gradient to pass to the previous layer.
        """
        # Compute the ReLU derivative: 1 for X > 0, otherwise 0
        relu_derivative = (self.X > 0).float()
        
        # Multiply incoming delta with the ReLU derivative to get delta_next
        delta_next = delta * relu_derivative
        return delta_next

""" Dropout Layer """

class DropoutLayer(nn.Module):
    def __init__(self, dropout_rate):
        """
        Initialize the Dropout layer.
        
        Args:
            dropout_rate: The probability of dropping a neuron.
        """
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs):
        """
        Apply Dropout during training.
        Automatically disabled during evaluation.
        
        Args:
            inputs: Tensor of any shape, the input activations.
        Returns:
            out: Tensor of the same shape as inputs, with dropout applied in training mode.
        """
        if self.training:
            # Create a dropout mask with probability (1 - dropout_rate)
            self.mask = (torch.rand_like(inputs) > self.dropout_rate).float()
            # Scale the outputs by (1 / (1 - dropout_rate)) to maintain expected value
            out = inputs * self.mask / (1 - self.dropout_rate)
        else:
            # During evaluation, dropout is not applied, so we pass inputs as they are
            out = inputs
        return out

    def backward(self, dout):
        """
        Perform the backward pass for (inverted) dropout.
        
        Args:
            dout: Upstream gradients of any shape.
        Returns:
            dout_next: Gradient with respect to the input x.
        """
        if self.training and self.mask is not None:
            # Only pass gradients where mask is 1, scaling by (1 - dropout_rate)
            dout_next = dout * self.mask / (1 - self.dropout_rate)
        else:
            # If not training, pass dout directly
            dout_next = dout
        return dout_next
