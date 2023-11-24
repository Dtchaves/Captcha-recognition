import torch
from torch import nn
from torchsummary import summary


class VGG16(nn.Module):
  def __init__(self,in_channels = 1,num_classes = 37):
    
    super(VGG16,self).__init__()
    self.in_channels = in_channels
    self.conv_layers = self.create_conv_layers()
    self.lin_layer = nn.Sequential(
      
      nn.Linear(512,4096),
      nn.ReLU(),
      nn.Dropout(p = 0.5),
      
      nn.Linear(4096,4096),
      nn.ReLU(),
      nn.Dropout(p = 0.5),
      
      nn.Linear(4096,num_classes),
    )
  
  def create_conv_layers(self):
    layers = []
    
    architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,'M']
    
    in_channels = self.in_channels
    
    for x in architecture:

      if type(x) == int:
        
        out_channels = x 
        
        layers += [nn.Conv2d(in_channels,out_channels, kernel_size= 3, stride = 1, padding = 1),
                  nn.BatchNorm2d(x),
                  nn.ReLU()]

        in_channels = x
        
      else:
        if x == "C":
          layers += [nn.MaxPool2d(kernel_size= 2, stride = 1)]
        else:
          layers += [nn.MaxPool2d(kernel_size= 2, stride = 2)]

        
    return nn.Sequential(*layers)
  
  def forward(self, x):
    x = self.conv_layers(x)
    #x = x.reshape(x.shape[0],-1)
    #x = self.lin_layer(x)
    return x
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}\n")
    
    print(f"Rodando VGG16\n")
    cnn_model = VGG16().to(device)
    summary(cnn_model, (1, 50, 32))
    