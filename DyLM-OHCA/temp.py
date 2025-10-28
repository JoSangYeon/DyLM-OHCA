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
    data_path = os.path.join('.', 'decoder_test_samples.csv')
    df = pd.read_csv(data_path)

    print(len(df.id.unique()))
    files_id = df.id.unique()
    new_data = pd.DataFrame()
    for fid in tqdm(files_id[:5120]):
        temp = df[df.id == fid]
        new_data = pd.concat([new_data, temp.iloc[-1:, :]], axis=0)

    # print(len(new_data.id.unique()))

    new_data.to_csv('decoder_based_data.csv', index=False)
    print(new_data)

    # main_device, device_ids = set_device(main_device_num=0, using_device_num=1)
    
    # # args
    # model_link = 'beomi/KcELECTRA-base-v2022' #'beomi/kcbert-base'
    
    # # epochs = 15
    # batch_size = 512 + 256
    # class_num = 2
    # max_length = 256 # 384
    # padding = 'max_length'
    
    # # Datasets
    # # train_path = "train_data.csv"
    # # valid_path = "valid_data.csv"
    # # test_path = "test_data.csv"
    # train_path = os.path.join('..', 'train_json_audio_data_end_full.csv')
    # valid_path = os.path.join('..', 'valid_json_audio_data_end_full.csv')
    # test_path = os.path.join('..', 'test_json_audio_data_end_full.csv')

    # train_data = pd.read_csv(train_path)
    # valid_data = pd.read_csv(valid_path)
    # test_data = pd.read_csv(test_path)

    # ## your Data Pre-Processing
    # print('init Data >>>')
    # print('\ttrain data :', train_data.shape)
    # print('\tvalid data :', valid_data.shape)
    # print('\tinit test data :', test_data.shape)

    # train_data = train_data.dropna(axis=0)
    # train_data = train_data.reset_index(drop=True)
    # valid_data = valid_data.dropna(axis=0)
    # valid_data = valid_data.reset_index(drop=True)
    # test_data = test_data.dropna(axis=0)
    # test_data = test_data.reset_index(drop=True)

    # print('Drop nan >>>')
    # print('\ttrain data :', train_data.shape)
    # print('\tvalid data :', valid_data.shape)
    # print('\ttest data :', test_data.shape)
    
    # for data in [train_data, valid_data, test_data]:
    #     cnt = []
    #     for end in data.endAt:
    #         cnt.append(end)
        
    #     print("평균 : {}".format(np.mean(cnt)))
    #     print("표편 : {}".format(np.std(cnt)))
    #     print("최대 : {}".format(np.max(cnt)))
    #     print("최소 : {}".format(np.min(cnt)))
    #     print()
    

        
if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=0 python test.py