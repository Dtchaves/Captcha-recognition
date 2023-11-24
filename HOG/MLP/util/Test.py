import matplotlib.pyplot as plt
import os
import torch
import sklearn.metrics as metrics
import pandas as pd
class Test:
    def __init__(self,model,train_loader,val_loader,model_name,path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.path = path
    
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
        save_path = os.path.join(self.path, name)
        plt.savefig(save_path)

    def Get_Y(self,device):
        y_true = [labels for _ , labels in self.val_loader]
        y_pred = []
        with torch.no_grad():
                for img, labels in self.val_loader:

                    img, labels = img.to(device), labels.to(device)

                    pred = self.model(img)
                    _, pred = torch.max(self.model(img), 1)
                    y_pred.append(pred)
                        
        y_true = torch.cat(y_true).to('cpu')
        y_pred = torch.cat(y_pred).to('cpu')
        print(type(y_true))
        print(type(y_pred))
        return y_true,y_pred
    
    def fit(self):
        self.model.eval()
        y_true,y_pred = self.Get_Y()
        self.metric(y_true=y_true, y_pred=y_pred)
        classification_report = metrics.classification_report(y_true, y_pred,target_names=None)
        return classification_report
    