import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


path = '/home/diogo/Documentos/final_icv/Dataset/'
original = os.path.join(path,'Original/')
train_folder = os.path.join(original,'treinamento/')
test_folder = os.path.join(original,'teste/')
validation_folder = os.path.join(original,'validacao/')
labels = os.path.join(original,'labels10k/')

cortado = os.path.join(path,'Cortado/')
train_folder_new  = os.path.join(cortado,'treinamento/')
test_folder_New = os.path.join(cortado,'teste/')
validation_folder_new  = os.path.join(cortado,'validacao/')
labels_new  = os.path.join(cortado,'labels10k/')

folders = [train_folder,test_folder,validation_folder]
folders_New = [train_folder_new,test_folder_New,validation_folder_new]


for fo in range(3):
    for filename in os.listdir(folders[fo]):
        
        label_file_name = os.path.join(labels, filename.replace('.jpg', '.txt'))
        with open(label_file_name,'r') as file:
            label = file.read()
            
        label = str(label)
        label = label.replace('\n', '')
        
        image_path = os.path.join(folders[fo], filename)
        image = io.imread(image_path,as_gray=True)
        size = int(image.shape[1]/6)
        
        for i in range(6):
            
            cropped_img = image[:,(i*size):((1+i)*size)]
            l = label[i]
            
            save_name = filename.replace('.jpg', '') + '_' + str(i)
            
            io.imsave(os.path.join(folders_New[fo],(save_name + '.jpg')),(cropped_img * 255).astype(np.uint8))
            
            with open(os.path.join(labels_new,(save_name + '.txt')),'w') as file:
                file.write(l)
            
