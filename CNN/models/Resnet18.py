import torch
from torch import nn
from torchsummary import summary


class Block_Resnet18(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample = None,stride = 1):
        self.stride = stride
        super(Block_Resnet18,self).__init__()
        self.block_layers = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size = 3,stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size = 3,stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
        )
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x
        x = self.block_layers(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        return self.relu(x + identity)
    
class Resnet18(nn.Module):
    def __init__(self,img_channels = 1,num_classes = 37):
        super(Resnet18,self).__init__()
        self.in_channels = img_channels;
        self.initial_layers = nn.Sequential(
            nn.Conv2d(img_channels,64, kernel_size = 7,stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride = 2,padding = 1)
        )
        self.in_channels = 64
        self.block_layers = self.create_block_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)

      
    def create_block_layers(self):        
        layers = []
        architecture = [(2,64,1),(2,128,2),(2,256,2),(2,512,2)]
            
        for num_blocks,out_channels,stride in architecture:
            identity_downsample = None
                
            if self.in_channels != out_channels:

                identity_downsample = nn.Sequential( 
                                            nn.Conv2d(self.in_channels,out_channels, kernel_size = 1,stride = stride),
                                            nn.BatchNorm2d(out_channels),
                                            )

               
                
            layers.append(Block_Resnet18(self.in_channels,out_channels,identity_downsample=identity_downsample,stride=stride))
            self.in_channels = out_channels;
                    
            for num in range(num_blocks - 1):
                    
                layers.append(Block_Resnet18(self.in_channels,out_channels,stride=1))
                


            

                


        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.block_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}\n")
    
    print(f"Rodando Resnet34\n")
    cnn_model = Resnet34().to(device)
    summary(cnn_model, (1, 50, 32))
    