import torch

class CosineLayer(torch.nn.Module):
    """
    Cosine input layer for solving the burnup problem.
    a_i * cos(Im(\lambda_j)*t) * exp(Re(\lambda_j)*t)
    """
    def __init__(self, input_dim, output_dim, weight):
        super(CosineLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.output_dim = output_dim
        self.weight.data = weight
        self.eval() # fixes the parameters

    def forward(self, x):
        m = self.output_dim
        real = self.weight.data.flatten()[:m]
        imag = self.weight.data.flatten()[m:]
        return torch.cos(imag * x) * torch.exp(real * x)


class SineLayer(torch.nn.Module):
    """
    Sine input layer for solving the burnup problem.
    b_i * sin(Im(\lambda_j)*t) * exp(Re(\lambda_j)*t)
    """
    def __init__(self, input_dim, output_dim, weight):
        super(SineLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.output_dim = output_dim
        self.weight.data = weight
        self.eval() # fixes the parameters

    def forward(self, x):
        m = self.output_dim
        real = self.weight.data.flatten()[:m]
        imag = self.weight.data.flatten()[m:]
        return torch.sin(imag * x) * torch.exp(real * x)


class LinearLayer(torch.nn.Module):
    """
    Linear layer implemented manually with ability to use float64 datatypes.
    """
    def __init__(self, in_features, out_features, dtype):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Initialize weight and bias tensors
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data = torch.randn((out_features, in_features), dtype=dtype)
        # self.bias = torch.zeros(out_features)

        # Initialize gradients
        self.weight_grad = None
        # self.bias_grad = None

    def forward(self, x):
        self.x = x  # Save the input for the backward pass
        # print(x.type(), self.weight.t().type())
        out = torch.matmul(x.to(self.dtype), self.weight.t().to(self.dtype))# + self.bias
        return out

    def backward(self, grad_output, learning_rate):
        # Compute gradients
        self.weight_grad = torch.matmul(grad_output.t(), self.x)
        # self.bias_grad = torch.sum(grad_output, dim=0)

        # Update weights and biases
        self.weight -= learning_rate * self.weight_grad
        # self.bias -= learning_rate * self.bias_grad

        # Propagate the gradients to the previous layer
        grad_input = torch.matmul(grad_output, self.weight)

        return grad_input
    