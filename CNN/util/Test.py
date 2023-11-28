import matplotlib.pyplot as plt
import seaborn as sns

import os
import torch
import sklearn.metrics as metrics
import pandas as pd

class Test:
    def __init__(self,model,test_loader,model_name,path_metric,path_n):
        self.model = model
        self.test_loader = test_loader
        self.model_name = model_name
        self.path_metric = path_metric
        self.path_n = path_n

    
    def metric(self,y_true, y_pred):
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred , average= 'macro')
        recall = metrics.recall_score(y_true, y_pred, average= 'macro')
        f1 = metrics.f1_score(y_true, y_pred,average= 'macro')
        report_table = {
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1]
        }

        report_table_df = pd.DataFrame(report_table)

        fig, ax = plt.subplots(figsize=(20, 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=report_table_df.values, colLabels=report_table_df.columns, cellLoc='center', loc='center')
        table.scale(1,2)

        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(weight='bold')
        name = f"Metrics_" + self.model_name
        save_path = os.path.join(self.path_metric, name)
        plt.savefig(save_path)
        

    def Get_Y(self,device):
        y_true = [labels for _ , labels in self.test_loader]
        y_pred = []
        with torch.no_grad():
                for img, labels in self.test_loader:

                    img, labels = img.to(device), labels.to(device)

                    pred = self.model(img)
                    _, pred = torch.max(self.model(img), 1)
                    y_pred.append(pred)
                    

                    
                    

        y_true = torch.cat(y_true).to('cpu')
        y_pred = torch.cat(y_pred).to('cpu')
        print(type(y_true))
        print(type(y_pred))
        return y_true,y_pred

    
    def Get_Accuracy_per_size(self,y_true,y_pred):
        prev =(y_pred == y_true)
        PM_1 = 0
        PM_2 = 0
        PM_3 = 0
        PM_4 = 0
        PM_5 = 0
        PM_6 = 0
        for i in range(0,len(prev),6):
            acertos = 0
            for j in range(6):
                if prev[i+j]:
                    acertos = acertos + 1
                    
            if(acertos == 6):
                PM_1 = PM_1 + 1
                PM_2 = PM_2 + 1
                PM_3 = PM_3 + 1
                PM_4 = PM_4 + 1
                PM_5 = PM_5 + 1
                PM_6 = PM_6 + 1
                
            if(acertos == 5):
                PM_1 = PM_1 + 1
                PM_2 = PM_2 + 1
                PM_3 = PM_3 + 1
                PM_4 = PM_4 + 1
                PM_5 = PM_5 + 1
            
            if(acertos == 4):
                PM_1 = PM_1 + 1
                PM_2 = PM_2 + 1
                PM_3 = PM_3 + 1
                PM_4 = PM_4 + 1
                
            if(acertos == 3):
                PM_1 = PM_1 + 1
                PM_2 = PM_2 + 1
                PM_3 = PM_3 + 1
                
            if(acertos == 2):
                PM_1 = PM_1 + 1
                PM_2 = PM_2 + 1
                
            if(acertos == 1):
                PM_1 = PM_1 + 1
                
        size = (len(prev)/6)
        y = [(PM_1/size),(PM_2/size),(PM_3/size),(PM_4/size),(PM_5/size),(PM_6/size)]
        x = [1,2,3,4,5,6]
        
        plt.figure(figsize=(14, 5)) 
        plt.plot(x, y, color="Blue", linewidth=2, marker='o', markersize=8)
        
        plt.grid(True, alpha=0.6)
        plt.gca().spines['top'].set_linewidth(0)
        plt.gca().spines['bottom'].set_linewidth(0.4)
        plt.gca().spines['left'].set_linewidth(0.4)
        plt.gca().spines['right'].set_linewidth(0)
        
        plt.xlabel("\nNúmero mínimo de caracteres reconhecidos por captcha", fontsize=9)
        plt.ylabel("Taxa de Reconhecimento\n", fontsize=9)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        plt.title("Resultado "+ self.model_name + "\n\n", fontsize=10)
        plt.gca().set_axisbelow(True)

        name = f"Acertos_por_quantidade_" + self.model_name
        save_path = os.path.join(self.path_n, name + ".png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    
    def fit(self,device):
        self.model.eval()
        y_true,y_pred = self.Get_Y(device)
        self.Get_Accuracy_per_size(y_true,y_pred)
        self.metric(y_true=y_true, y_pred=y_pred)
        classification_report = metrics.classification_report(y_true, y_pred,target_names=None)
        return classification_report
    