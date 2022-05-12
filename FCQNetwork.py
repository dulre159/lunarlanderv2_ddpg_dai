from torch import nn

class FCQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCQNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits