import torch
from torch import nn
from torchsummary import summary

class MLP(nn.Module):
  def __init__(self):
      super().__init__()
      
      self.Layers = nn.Sequential(
          nn.Linear(3780, 3780),
          nn.Dropout(p=0.3),
          nn.ReLU(),
          nn.Linear(3780, 3780),
          nn.Dropout(p=0.3),
          nn.ReLU(),
          nn.Linear(3780, 3780),
          nn.Dropout(p=0.3),
          nn.ReLU(),
          nn.Linear(3780, 37),
      )

  def forward(self, x):
    x = self.Layers(x)
    return nn.functional.softmax(x, dim=-1)
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}\n")
    
    print(f"Rodando MLP\n")
    model = MLP().to(device)
    summary(model, (3780,))