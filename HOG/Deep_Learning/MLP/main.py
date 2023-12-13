import torch

from models.MLP import *
from util.Test import *
from util.Trainer import *
from util.dataloaders.Dataloader import CaptchaDataloader
from torch.utils.data import DataLoader
from torchvision import transforms


#Load data
batch_size = 64


train_data = 
val_data = 
test_data = 


train_loader = 
val_loader = 
test_loader = 




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")

model = MLP().to(device)
trainer = Trainer(model=model,train_loader=train_loader,val_loader=val_loader,model_name="MLP",path_par='',path_loss='')
trainer.run(device=device,epochs=10)

test = Test(model,train_loader,val_loader,"MLP",)
classification_report = test.fit(device=device)
print(classification_report)
