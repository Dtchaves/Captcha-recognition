import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from tqdm import tqdm



class Trainer:
    def __init__(self,model,train_loader,val_loader,model_name,path_par,path_loss):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.path_par = path_par
        self.path_loss = path_loss

        
    def plot_loss(self,train_losses, val_losses,epoch):
        fig = plt.figure(figsize=(13,5))
        ax = fig.gca()
        plt.ion()
        ax.plot(train_losses, label="Train loss", color = "blue")
        ax.plot(val_losses, label="Validation loss", color = "orange")
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Loss vs iterations", fontsize="16");
        name = self.model_name
        save_path = os.path.join(self.path_loss, name)
        plt.savefig(save_path)
        
    def save_models(self, epoch,best_values):
        name = f"epoch{epoch}_" + self.model_name
        save_path = os.path.join(self.path_par, name)
        torch.save(best_values, save_path)
        
    def run(self,epochs,device):
        cnn_loss_func = nn.CrossEntropyLoss()
        cnn_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        conv_train_losses = []
        conv_val_losses = []
        
        print("\n\n ----- STARTING TRAINING -----\n\n")
        for t in range(epochs):
            
            train_loss = 0.0
            val_loss = 0.0
            actual_loss = 0.0
            save_epo = 0.0

            for img, label in tqdm(self.train_loader, desc=f'TRAINING EPOCH {t}/{epochs-1}',dynamic_ncols=True,colour="BLUE",):
                cnn_optimizer.zero_grad()
                
                img,label = img.to(device) ,label.to(device)
                
                pred = self.model(img)
                loss = cnn_loss_func(pred, label)
                loss.backward()
                cnn_optimizer.step()
                
                train_loss += loss.item()
                
            train_loss = train_loss/len(self.train_loader)
            conv_train_losses.append(train_loss)
            
            with torch.no_grad():
                for img, labels in self.val_loader:
                    
                    img, labels = img.to(device), labels.to(device)

                    pred = self.model(img)
                    loss = cnn_loss_func(pred, labels)
                    val_loss += loss.item()
                
            val_loss = val_loss / len(self.val_loader)
            conv_val_losses.append(val_loss)
            if actual_loss < val_loss:
                actual_loss = val_loss
                save_epo = t
                best_values = self.model.state_dict()
            if t % 10 == 0:
                print(f"Epoch: {t} \nTrain Loss: {train_loss}\nTest Loss: {val_loss}\n")
            if t != 0:
                self.plot_loss(conv_train_losses,conv_val_losses,t)         
                    
        self.save_models(save_epo,best_values)