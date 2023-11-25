from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

import numpy as np
from skimage import io
import torch
import os

class CaptchaDataloader(Dataset):
    def __init__(self, root_dir, split='treinamento', transform=None,label_dir = 'labels10k'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_dir = label_dir
        
        self.img_dir = os.path.join(root_dir,split)
        self.lbl_dir = os.path.join(root_dir,label_dir)

        self.imgs_files = os.listdir(self.img_dir)
        self.lbls_files = [f.replace('.jpg', '.txt') for f in self.imgs_files]
        
    def __len__(self):
        return(len(self.imgs_files))

    def __getitem__(self, idx) :
        
        img_name = os.path.join(self.img_dir, self.imgs_files[idx])
        lbl_name = os.path.join(self.lbl_dir, self.lbls_files[idx])

        image = io.imread(img_name,as_gray=True)
        
        with open(lbl_name,'r') as file:
            label_str = file.read()
                    

        image = torch.from_numpy(image)
        image = image.to(torch.float32)
        image = image[None,:,:]


        if self.transform:
            image = self.transform(image)



        label_str = str(label_str)
        label_str = label_str.replace('\n', '')
        
        label = [ord(char) - 27 if ord(char) == 63 else ord(char) - 48 if 48 <= ord(char) <= 57 else ord(char) - 55 for char in label_str]
        label = int(label[0])

        #CrossEntropy nao usa hot
        label_hot = np.zeros(37)
        label_hot[label] = 1
        label_hot = torch.tensor(label_hot)
        
        return image,label
    
    
    

if __name__ == "__main__":
    batch_size = 64
    
    resize_transform = transforms.Resize((50,32))

    teste =  CaptchaDataloader(split='treinamento',transform= resize_transform,root_dir='/home/diogo/Documentos/final_icv/Dataset/Cortado')
    teste = DataLoader(dataset=teste, batch_size=batch_size, shuffle=True)
    
    for batch in teste:
        inputs, labels = batch
        
        item = inputs.shape

        print("Input Shape:",inputs[0].shape)
        print("Label:",labels[0])
        plt.imshow(inputs[0,0,:,:], cmap="gray")
        plt.show()

