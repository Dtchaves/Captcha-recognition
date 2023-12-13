from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

import numpy as np
import math
import skimage
from skimage import io
import torch
import os


class CaptchaDataloader(Dataset):
    def _init_(self, root_dir, split='treinamento', transform=None,label_dir = 'labels10k'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_dir = label_dir
        
        self.img_dir = os.path.join(root_dir,split)
        self.lbl_dir = os.path.join(root_dir,label_dir)

        self.imgs_files = os.listdir(self.img_dir)
        self.lbls_files = [f.replace('.jpg', '.txt') for f in self.imgs_files]
        
    def _len_(self):
        return(len(self.imgs_files))

    def extract_hog_features(self,img):
        # image = skimage.io.imread('teste/treinamento/000001.jpg',as_gray=True)
        img = skimage.transform.resize(img, (128,64))
        # print(len(img[1]))
        mag = []
        theta = []

        for i in range(128):
            magnitudeArray = []
            angleArray = []

            for j in range(64):
                # Condition for axis 0
                if j-1 <= 0 or j+1 >= 64:
                    if j-1 <= 0:
                        # Condition if the first element
                        Gx = img[i, j+1] - 0
                    elif j + 1 >= 64:
                        Gx = 0 - img[i, j-1]
                # Condition for the first element
                else:
                    Gx = img[i, j+1] - img[i, j-1]

                # Condition for axis 1
                if i-1 <= 0 or i+1 >= 128:
                    if i-1 <= 0:
                        Gy = 0 - img[i+1, j]
                    elif i +1 >= 128:
                        Gy = img[i-1, j] - 0
                else:
                    # print(i, j)
                    Gy = img[i-1, j] - img[i+1, j]

                # Calculating magnitude
                # print(Gx)
                magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
                magnitudeArray.append(round(magnitude, 9))

                # Calculating angle
                if Gx == 0:
                    angle = np.degrees(0.0)
                else:
                    angle = np.degrees(np.abs(np.arctan(Gy / Gx)))
                angleArray.append(round(angle, 9))

            mag.append(magnitudeArray)
            theta.append(angleArray)

        mag = np.array(mag)
        theta = np.array(theta)

        row, col = mag.shape
        features = []

        for i in range(0, row, 8):
            for j in range(0, col, 8):
                curr_mag = mag[i:i+8, j:j+8]
                curr_theta = theta[i:i+8, j:j+8]
                window_row, window_col = curr_mag.shape
                histogram = np.zeros(9)

                for k in range (window_row):
                    for l in range (window_col):
                        first_bin = int(curr_theta[k, l] / 20)
                        second_bin = (math.ceil(curr_theta[k, l] / 20)) % 9

                        if first_bin == second_bin:
                            histogram[first_bin] += curr_mag[k, l]
                        else:
                            histogram[first_bin] += curr_mag[k, l] * (1 - (curr_theta[k, l] - first_bin * 20) / 20)
                            histogram[second_bin] += curr_mag[k, l] * ((curr_theta[k, l] - first_bin * 20) / 20)

                features += histogram.tolist()

        return features


    def _getitem_(self, idx) :
        
        img_name = os.path.join(self.img_dir, self.imgs_files[idx])
        lbl_name = os.path.join(self.lbl_dir, self.lbls_files[idx])

        image = io.imread(img_name,as_gray=True)
        
        with open(lbl_name,'r') as file:
            label_str = file.read()
                    
        hog_image = self.extract_hog_features(image)
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
    
    
    

if __name__ == "_main_":
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