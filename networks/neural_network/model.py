import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, number_of_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.number_of_layers = number_of_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))  

        for _ in range(number_of_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
        
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.net(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
