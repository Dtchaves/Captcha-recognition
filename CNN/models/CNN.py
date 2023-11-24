import torch
from torch import nn
from torchsummary import summary

class CNN_net(nn.Module):
  def __init__(self):
      super().__init__()
      self.convlayers = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(16, 32, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

      )

      self.linearlayers = nn.Sequential(
          nn.Linear(2112, 1056),
          nn.ReLU(),
          nn.Linear(1056, 528),
          nn.ReLU(),
          nn.Linear(528, 264),
          nn.ReLU(),
          nn.Linear(264, 37),
      )

  def forward(self, x):
      x = self.convlayers(x)
      x = torch.flatten(x, 1)
      x = self.linearlayers(x)
      return x
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}\n")
    
    print(f"Rodando CNN\n")
    cnn_model = CNN_net().to(device)
    summary(cnn_model, (1, 50, 30))
    