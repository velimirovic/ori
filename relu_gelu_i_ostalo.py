import torch
import torch.nn as nn
import math

# 1. KREIRANJE CUSTOM GELU AKTIVACIONE FUNKCIJE
class CustomGELU(nn.Module):
    def __init__(self):
        super(CustomGELU, self).__init__()
    
    def forward(self, x):
        # GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x³)))
        
        # Konstante
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)  # √(2/π)
        coeff = 0.044715
        
        # Računamo deo unutar tanh funkcije
        inner = sqrt_2_over_pi * (x + coeff * torch.pow(x, 3))
        
        # Tanh funkcija (možemo koristiti ugrađenu ili implementirati sami)
        tanh_part = torch.tanh(inner)
        
        # Finalna GELU formula
        gelu_result = 0.5 * x * (1 + tanh_part)
        
        return gelu_result

class CustomMish(nn.Module):
    def __init__(self):
        super(CustomMish, self).__init__()
    
    def forward(self, x):
        # Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        return x * torch.tanh(torch.softplus(x))
    
class CustomLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(CustomLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        # LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
        return torch.maximum(torch.zeros_like(x), x) + self.negative_slope * torch.minimum(torch.zeros_like(x), x)
        # Ili kraće:
        # return torch.where(x > 0, x, self.negative_slope * x)

class CustomELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(CustomELU, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        # ELU(x) = x if x > 0, else alpha * (e^x - 1)
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
    
class CustomSELU(nn.Module):
    def __init__(self):
        super(CustomSELU, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    
    def forward(self, x):
        # SELU(x) = scale * ELU(x, alpha)
        return self.scale * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
    
class CustomHardSwish(nn.Module):
    def __init__(self):
        super(CustomHardSwish, self).__init__()
    
    def forward(self, x):
        # HardSwish(x) = x * ReLU6(x + 3) / 6
        return x * torch.clamp(x + 3, 0, 6) / 6
    
class CustomGELUTanh(nn.Module):
    def __init__(self):
        super(CustomGELUTanh, self).__init__()
    
    def forward(self, x):
        # GELU aproksimacija: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

class CustomGELUErf(nn.Module):
    def __init__(self):
        super(CustomGELUErf, self).__init__()
    
    def forward(self, x):
        # Tačna GELU formula: 0.5 * x * (1 + erf(x / √2))
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
    
class CustomFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(CustomFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
class CustomSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.velocity[i] = self.momentum * self.velocity[i] + param.grad
                param.data -= self.lr * self.velocity[i]
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()