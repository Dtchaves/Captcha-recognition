import torch
#teste
from models.CNN import *
from util.Test import *
from util.Trainer import *
from util.dataloaders.Dataloader import CaptchaDataloader
from torch.utils.data import DataLoader
from torchvision import transforms


#Load data
batch_size = 64


#To do
#transformation = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=,std=)
#]) 


train_data = CaptchaDataloader(split='treinamento',root_dir='/home/diogo/Documentos/final_icv/Dataset/Cortado')
val_data = CaptchaDataloader(split='validacao',root_dir='/home/diogo/Documentos/final_icv/Dataset/Cortado')
test_data = CaptchaDataloader(split='teste',root_dir='/home/diogo/Documentos/final_icv/Dataset/Cortado')





train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False)

test_loader



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")

cnn_model = CNN_net().to(device)
trainer = Trainer(model=cnn_model,train_loader=train_loader,val_loader=val_loader,model_name="CNN",path_par='/home/diogo/Documentos/final_icv/CNN/results/best_model',path_loss='/home/diogo/Documentos/final_icv/CNN/results/loss')
trainer.run(device=device,epochs=10)

test = Test(cnn_model,test_loader,"CNN",path='/home/diogo/Documentos/final_icv/CNN/results/metrics')
classification_report = test.fit(device=device)
print(classification_report)
