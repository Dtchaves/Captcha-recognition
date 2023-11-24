import torch
from torch import nn
from torchsummary import summary


class Block_Resnet34(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample = None, stride = 1):
        
        super(Block_Resnet34,self).__init__()
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

class Block_Resnet50(nn.Module):
    def __init__(self,in_channels,out_channels,identity_downsample = None, stride = 1,expansion = 4):
        
        super(Block_Resnet50,self).__init__()
        self.expansion = expansion
        self.block_layers = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size = 1,stride = 1, padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size = 3,stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels*expansion, kernel_size = 1,stride = 1, padding = 0),
            nn.BatchNorm2d(out_channels*expansion),
        )
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x
        x = self.block_layers(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        return self.relu(x + identity)
    
class Resnet(nn.Module):
    def __init__(self,type_res,img_channels = 3,num_classes = 10):
        super(Resnet,self).__init__()
        self.in_channels = img_channels;
        self.type_res = type_res
        self.initial_layers = nn.Sequential(
            nn.Conv2d(img_channels,64, kernel_size = 7,stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride = 2,padding = 1)
        )
        self.in_channels = 64
        self.block_layers = self.create_block_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if self.type_res == "Resnet50": self.fc = nn.Linear(512*4,num_classes)
        elif self.type_res == "Resnet34": self.fc = nn.Linear(512,num_classes)

      
    def create_block_layers(self):        
        layers = []
        architecture = [(3,64,1),(4,128,2),(3,256,2),(3,512,2)] # (blocks,channels,stride)
            
        for num_blocks,out_channels,stride in architecture:
            
            identity_downsample = None
                
            if self.type_res == "Resnet50": 
                if stride != 1 or self.in_channels != out_channels*4:
                    identity_downsample = nn.Sequential( 
                                                        nn.Conv2d(self.in_channels,out_channels*4, kernel_size = 1,stride = stride, padding = 0),
                                                        nn.BatchNorm2d(out_channels*4),
                                                        )
                    
                layers.append(Block_Resnet50(self.in_channels,out_channels,identity_downsample=identity_downsample, stride = stride))
                self.in_channels = out_channels*4;
                
            elif self.type_res == "Resnet34":
                if self.in_channels != out_channels:
                    identity_downsample = nn.Sequential( 
                                                        nn.Conv2d(self.in_channels,out_channels, kernel_size = 3,stride = stride, padding = 1),
                                                        nn.BatchNorm2d(out_channels),
                                                        )
                layers.append(Block_Resnet34(self.in_channels,out_channels,identity_downsample=identity_downsample, stride = stride))
                self.in_channels = out_channels;
                    
            for num in range(num_blocks - 1):
                    
                if self.type_res == "Resnet34":  layers.append(Block_Resnet34(self.in_channels,out_channels))
                elif self.type_res == "Resnet50":   layers.append(Block_Resnet50(self.in_channels,out_channels))

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
    cnn_model = Resnet(type_res= "Resnet34").to(device)
    summary(cnn_model, (3, 224, 224))
    
    print(f"Rodando Resnet50\n")
    cnn_model = Resnet(type_res= "Resnet50").to(device)
    summary(cnn_model, (3, 224, 224))
    