# https://github.com/HideOnHouse/TorchBase

import os
import glob
import wandb
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler

from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import (roc_auc_score, 
                             average_precision_score,
                             accuracy_score,  
                             recall_score,
                             precision_score,
                             f1_score,
                             brier_score_loss, # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
                             confusion_matrix,)
from imblearn.metrics import specificity_score

import warnings
warnings.filterwarnings(action='ignore')

SEED = 17
# random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_sample_result(file_path):
    print(file_path)
    df = pd.read_csv(file_path)
    
    # metric = {
    #     'id': [],
    #     'mean_prob': [],
    #     'label': [],
    # }
    # id_list = df.id.unique()
    # print('>>> Calc prob as sample units')
    # for id in tqdm(id_list):
    #     temp = df[df.id==id]
        
    #     prob_list = temp.predicted_probas.values
    #     mean_prob = prob_list.mean()
    #     label = temp.labels.values[0]
        
    #     metric['id'].append(id)
    #     metric['mean_prob'].append(mean_prob)
    #     metric['label'].append(label)
        
    # metric_df = pd.DataFrame(metric)
    
    metric_df = df.groupby('id').mean()
    
    predicted_proba = metric_df.predicted_probas
    label = metric_df.labels
    
    print('>>> Thresholding : recall-based')
    result = {
        'THRESHOLD' : [],
        'RECALL' : [],
        'THACC' : [],
        'AUROC' : [],
        'AUPRC' : [],
        'PRECISION' : [],
        'SPECIFICITY' : [],
        'F1': [],
        'BRIER': [],
        }
    for threshold in tqdm(np.linspace(0, 1, 512)):
        predicted_label = np.where(predicted_proba >= threshold , 1, 0)
        
        result['THRESHOLD'].append(round(threshold, 5))
        result['THACC'].append(round(accuracy_score(label, predicted_label), 5))
        result['AUROC'].append(round(roc_auc_score(label, predicted_proba), 5))
        result['AUPRC'].append(round(average_precision_score(label, predicted_proba), 5))
        result['RECALL'].append(round(recall_score(label, predicted_label), 5))
        result['PRECISION'].append(round(precision_score(label, predicted_label), 5))
        result['SPECIFICITY'].append(round(specificity_score(label, predicted_label), 5))
        result['F1'].append(round(f1_score(label, predicted_label), 5))
        result['BRIER'].append(round(brier_score_loss(label, predicted_proba), 5))
    
    result_df = pd.DataFrame(result)
    return result_df

def main():
    check_epoch = [1]
    check_iter = [i * 5120 for i in range(1, 12)] + [60154]
    
    # print(metric_list)
    for e in check_epoch:
        for it in check_iter:
            file_path = os.path.join('models', 
                                     'Only_Text_stacked_e3_bs48', 
                                     f'value_test_checkpoint_{e}_{it}.csv')
            save_path = os.path.dirname(file_path)
            
            result_df = get_sample_result(file_path)
            result_df.to_csv(os.path.join(save_path, 
                                          f'result_checkpoint_{e}_{it}.csv'),
                             index=False)
            print() 
        
if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=7 python test_sample_unit.py
    

def show_nan(data):
    df = data.iloc[:, 5:]
    try:
        for i in range(len(df)):
            total_text = ''
            text_values = df.iloc[i].values.reshape(-1, 2)
            for spk_id, t in text_values:
                if pd.isna(spk_id) or pd.isna(t):
                    break
                text = f'[SPK{int(spk_id)}]' + ' ' + t
                total_text += text + ' '
    except:
        print(data.id[i])
        print(total_text)