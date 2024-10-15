import torch
import torch.nn as nn

class NeuralNet(nn.Module):  # Fixed typo: 'Modele' to 'Module'
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  # Initialize the parent class inside __init__
        self.l1 = nn.Linear(input_size, hidden_size)  # Fixed typo: 'linear' to 'Linear'
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Fixed typo: 'linear' to 'Linear'
        self.l3 = nn.Linear(hidden_size, num_classes)  # Fixed typo: 'linear' to 'Linear'
        self.relu = nn.ReLU()  # Fixed typo: 'RelU' to 'ReLU'

    def forward(self, x):  # Fixed typo: 'forwad' to 'forward'
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
