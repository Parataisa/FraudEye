import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, number_of_layers):
        super(Net, self).__init__()
        self.input_size = input_size

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        if hidden_size > 1:
            layers.append(nn.BatchNorm1d(hidden_size))
        
        for _ in range(number_of_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if hidden_size > 1:
                layers.append(nn.BatchNorm1d(hidden_size))
        
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.net(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)