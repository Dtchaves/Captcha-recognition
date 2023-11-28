import torch

from colorama import Fore, Style

from models.CNN import *
from models.Resnet34 import *
from models.Resnet18 import *
from models.Vgg16 import *
from util.Test import *
from util.Trainer import *

from util.dataloaders.Dataloader import CaptchaDataloader
from torch.utils.data import DataLoader
from torchvision import transforms


batch_size = 64


resize_transform = transforms.Resize((50,32), antialias=True)

train_data = CaptchaDataloader(split='treinamento',transform= resize_transform,root_dir='/scratch/diogochaves/Projetos/ICV/Dataset/Cortado')
val_data = CaptchaDataloader(split='validacao',transform= resize_transform,root_dir='/scratch/diogochaves/Projetos/ICV/Dataset/Cortado')
test_data = CaptchaDataloader(split='teste',transform= resize_transform,root_dir='/scratch/diogochaves/Projetos/ICV/Dataset/Cortado')





train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False)

test_loader


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nRodando modelo de RESNET34 na {device}")

cnn_model = Resnet34().to(device)
entrada = ''
while entrada != "Load" and entrada != "Train":
    entrada = input("\nLoad or Train?\n")

    if entrada == "Load":
        print(f"{Fore.BLUE}\n ----- STARTING LOADING -----\n{Style.RESET_ALL}")
        
        path_dic = '/scratch/diogochaves/Projetos/ICV/CNN/results/best_w/Best_w_RESNET34'
        dic = torch.load(path_dic)
        cnn_model.load_state_dict(dic)
        
        print(f"{Fore.GREEN}LOADING COMPLETED{Style.RESET_ALL}")
        
    elif entrada == "Train":
        trainer = Trainer(model=cnn_model,train_loader=train_loader,val_loader=val_loader,model_name="RESNET34",path_par='/scratch/diogochaves/Projetos/ICV/CNN/results/best_w',path_loss='/scratch/diogochaves/Projetos/ICV/CNN/results/loss')
        trainer.run(device=device,epochs=40)

print(f"{Fore.BLUE}\n ----- STARTING TESTING -----\n{Style.RESET_ALL}")

test = Test(cnn_model,test_loader,"RESNET34",path_metric='/scratch/diogochaves/Projetos/ICV/CNN/results/metrics',path_n='/scratch/diogochaves/Projetos/ICV/CNN/results/por_quantidade')
classification_report = test.fit(device=device)
print(classification_report)

print(f"{Fore.GREEN}TESTING COMPLETED{Style.RESET_ALL}")
