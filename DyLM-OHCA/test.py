# https://github.com/HideOnHouse/TorchBase

import os
import glob
import wandb
import pickle
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

from dataset import *
from learning import *
from model import *
# from inference import *
# from inference_switching_recall import *
from utils import DataParallelModel, DataParallelCriterion
from utils import set_device, calc_metric

import warnings
warnings.filterwarnings(action='ignore')

SEED = 42
# random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def main():
    main_device, device_ids = set_device(main_device_num=0, using_device_num=1)
    
    # args
    model_link = 'beomi/KcELECTRA-base-v2022' #'beomi/kcbert-base'
    
    # epochs = 15
    batch_size = 512 + 256
    class_num = 2
    max_length = 256 # 384
    padding = 'max_length'
    
    # Datasets
    # train_path = "train_data.csv"
    # valid_path = "valid_data.csv"
    # test_path = "test_data.csv"
    train_path = os.path.join('..', 'train_json_audio_data.csv')
    valid_path = os.path.join('..', 'valid_json_audio_data.csv')
    test_path = os.path.join('..', 'test_json_audio_data.csv')

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)

    ## your Data Pre-Processing
    print('init Data >>>')
    print('\ttrain data :', train_data.shape)
    print('\tvalid data :', valid_data.shape)
    print('\tinit test data :', test_data.shape)

    train_data = train_data.dropna(axis=0)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.dropna(axis=0)
    valid_data = valid_data.reset_index(drop=True)
    test_data = test_data.dropna(axis=0)
    test_data = test_data.reset_index(drop=True)

    print('Drop nan >>>')
    print('\ttrain data :', train_data.shape)
    print('\tvalid data :', valid_data.shape)
    print('\ttest data :', test_data.shape)
    
    ## Create Dataset and DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_link)
    train_dataset = MyDataset(train_data, 
                              tokenizer, 
                              max_length=max_length, 
                              padding=padding,
                              class_num=class_num)
    valid_dataset = MyDataset(valid_data,
                              tokenizer,
                              max_length=max_length,
                              padding=padding,
                              class_num=class_num)
    test_dataset = MyDataset(test_data,
                             tokenizer,
                             max_length=max_length,
                             padding=padding,
                             class_num=class_num)
    
    test_file_ids = test_data.id
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=RandomSampler(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    ## label_frequency
    train_label_frequency = (train_data.label == 1).sum() / len(train_data)
    valid_label_frequency = (valid_data.label == 1).sum() / len(valid_data)
    test_label_frequency = (test_data.label == 1).sum() / len(test_data)
    inference_label_frequency = 0.1906 # 적절한 threshold 정하기
    print("Label frequency of Train Data: {:6f}".format(train_label_frequency))
    print("Label frequency of Valid Data: {:6f}".format(valid_label_frequency))
    print("Label frequency of Test Data: {:6f}".format(test_label_frequency))
    print("Label frequency of Test Data: {:6f}".format(inference_label_frequency))
    
    ## testing & inference
    ckpt_path = os.path.join('.', 'checkpoint_2_392.tar') # chceck your path
    
    file_name = os.path.basename(ckpt_path).split('.')[0]
    model = Baseline(model_link=model_link, class_num=2)
    criterion = torch.nn.CrossEntropyLoss()
    
    ckpt = torch.load(ckpt_path, map_location=main_device)
    model.load_state_dict(ckpt['model_state_dict']); model.to(main_device)
    test_loss, test_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER), (predicted_probas, labels)= inference_evaluate(model, 
                                                                                                                                  main_device, 
                                                                                                                                  criterion, 
                                                                                                                                  inference_label_frequency, 
                                                                                                                                  test_loader)
    print("test loss : {:.6f}".format(test_loss))
    # print("test acc : {:.3f}".format(test_acc))
    print("test acc(th) : {:4f}".format(TH_ACC))
    print("test AUROC : {:.4f}".format(AUROC))
    print("test AUPRC : {:.4f}".format(AUPRC))
    print("test Recall : {:4f}".format(RECALL))    
    print("test Precision : {:.4f}".format(PRECISION))
    print("test F1_score : {:.4f}".format(F1))
    print("test Brier : {:4f}".format(BRIER))
    # print(f"TN: {CM[0][0]} | FP: {CM[0][1]} | FN: {CM[1][0]} | TP: {CM[1][1]}")
    prediction_values = pd.DataFrame({'id':test_file_ids,
                                      'predicted_probas':predicted_probas, 
                                      'predicted_labels':np.where(predicted_probas >= inference_label_frequency, 1, 0),
                                      'labels':labels})
    prediction_values.to_csv(os.path.join('.', f'개별_항목별_값.csv'), index=False)
    # result_df = calc_metric(predicted_probas, labels)
    # result_df.to_csv(os.path.join('.', f'TEST_{file_name}.csv'), index=False)
        
if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=0 python test.py