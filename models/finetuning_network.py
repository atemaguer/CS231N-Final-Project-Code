import torch

class FineTuningModel(torch.nn.Module):
    def __init__(self, encoder, input_dim, output_dim):
        super(FineTuningModel, self).__init__()
        self.encoder = encoder
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.encoder(x)
        out =  torch.mean(out, dim=[2, 3])
        return self.linear(out)
