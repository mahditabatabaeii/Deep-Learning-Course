# You are not allowed to import any other libraries or modules.

import torch

class SGD:
    def __init__(self, params, learning_rate):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.
        
        Args:
            params: List of model parameters to be updated.
            learning_rate: The learning rate for updating the parameters.
        """
        self.params = list(params)  # Model parameters to be updated
        self.learning_rate = learning_rate

    def step(self):
        """
        Perform a parameter update using Stochastic Gradient Descent (SGD).
        """
        # Loop over all parameters
        with torch.no_grad():  # Ensure gradients are not tracked
            for param in self.params:
                if param.grad is not None:  # Check if the parameter has a gradient
                    # Update the parameter: theta = theta - learning_rate * gradient
                    param -= self.learning_rate * param.grad

    def zero_grad(self):
        """
        Zero the gradients for all parameters.
        """
        # Loop over all parameters and zero out the gradient
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
