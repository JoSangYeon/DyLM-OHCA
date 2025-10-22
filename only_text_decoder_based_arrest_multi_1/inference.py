import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler

from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
from transformers import BertTokenizerFast
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

from dataset import *
from learning import *
from model import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

def setup(seed):
    # random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup(args.SEED)
    
    sentence_ps = args.sentence_ps #'moving_average'
    window_size = args.window_size
    
    # Define project
    project_name = f'NIA_119-GPT_MULTI_1'
    model_name = 'only_Text_GPT_MULTI_1'
    model_link = "kykim/gpt3-kor-small_based_on_gpt2" # 'skt/kogpt2-base-v2' #'beomi/kcbert-base'

    # args
    rank = 'cuda:0'
    epochs = 10
    batch_size = 12
    lr = 1e-5
    
    class_num = 3
    speaker_num = 4
    max_length = 768
    padding = 'max_length'
    save_term = 710
    
    # dataset
    train_path = os.path.join('..', 'NIA_text_dataset', 'train_json_audio_data_decoder_time.csv')
    valid_path = os.path.join('..', 'NIA_text_dataset', 'valid_json_audio_data_decoder_time.csv')
    test_path = os.path.join('..', 'NIA_text_dataset', 'test_json_audio_data_decoder_time.csv')
    # train_path = os.path.join('..', 'NIA_text_dataset', 'toy_data_json_audio_data_decoder_time.csv')
    # valid_path = os.path.join('..', 'NIA_text_dataset', 'toy_data_json_audio_data_decoder_time.csv')
    # test_path = os.path.join('..', 'NIA_text_dataset', 'toy_data_json_audio_data_decoder_time.csv')
    
    save_path = os.path.join('models', 'only_Text_GPT_MULTI_1_e10_bs12')  
    ckpt_path = os.path.join(save_path, 'checkpoint_8_710.tar')

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)
    valid_file_ids = valid_data.id
    test_file_ids = test_data.id

    ## your Data Pre-Processing
    print('init Data >>>')
    print('\tinit train data :', train_data.shape)
    print('\tinit valid data :', valid_data.shape)
    print('\tinit test data :', test_data.shape)

    # train_data = train_data.dropna(axis=0)
    train_data = train_data.reset_index(drop=True)
    # valid_data = valid_data.dropna(axis=0)
    valid_data = valid_data.reset_index(drop=True)
    # test_data = test_data.dropna(axis=0)
    test_data = test_data.reset_index(drop=True)

    print('\ttrain data :', train_data.shape)
    print('\tvalid data :', valid_data.shape)
    print('\ttest data :', test_data.shape)
    
    ## Create Dataset and DataLoader
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_link,bos_token='<s>', eos_token='</s>', 
    #                                               unk_token='<unk>',pad_token='<pad>', mask_token='<mask>')
    tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
    special_tokens_dict = {'additional_special_tokens': [f'[SPK{n}]' for n in range(speaker_num)]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    train_dataset = MyDataset(train_data, 
                              tokenizer, 
                              max_length=max_length, 
                              padding=padding,
                              speaker_num=speaker_num,
                              class_num=class_num)
    valid_dataset = MyDataset(valid_data,
                              tokenizer,
                              max_length=max_length,
                              padding=padding,
                              speaker_num=speaker_num,
                              class_num=class_num)
    test_dataset = MyDataset(test_data,
                             tokenizer,
                             max_length=max_length,
                             padding=padding,
                             speaker_num=speaker_num,
                             class_num=class_num)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    ## label_frequency
    train_label_frequency = (train_data.label1 == 1).sum() / len(train_data)
    valid_label_frequency = (valid_data.label1 == 1).sum() / len(valid_data)
    test_label_frequency = (test_data.label1 == 1).sum() / len(test_data)
    print("Label frequency of Train Data: {:6f}".format(train_label_frequency))
    print("Label frequency of Valid Data: {:6f}".format(valid_label_frequency))
    print("Label frequency of Test Data: {:6f}".format(test_label_frequency))
    
    # modeling
    model = GPT_Baseline.from_pretrained(model_link, class_num=class_num,
                                         pad_token_id=pad_token_id, cls_token_id=cls_token_id, sep_token_id=sep_token_id)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(rank)

    # optimizer = optim.AdamW([{'params': model.module.electra.parameters(),'lr': electra_lr},
    #                          {'params': model.module.classifier.parameters(),'lr': cls_lr}],
    #                         eps=1e-8)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # iter_len = len(train_loader)
    # num_training_steps = iter_len * epochs
    # num_warmup_steps = int(0.15 * num_training_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=num_warmup_steps,
    #                                             num_training_steps=num_training_steps)

    print(f"{ckpt_path} >>> ")
    file_name = os.path.basename(ckpt_path).split('.')[0]
    model = GPT_Baseline.from_pretrained(model_link, class_num=class_num, pad_token_id=pad_token_id,
                                         cls_token_id=cls_token_id, sep_token_id=sep_token_id,
                                         sentence_ps=sentence_ps, window_size=window_size)
    model.resize_token_embeddings(len(tokenizer))
    ckpt = torch.load(ckpt_path, map_location=rank)
    model.load_state_dict(ckpt['model_state_dict']); model.to(rank)
    model.sentence_ps = sentence_ps
    model.window_size = window_size
    
    ((predicted_probas, labels_), 
     (file_ids_list, predicted_token_proba_0, 
      predicted_token_proba_1, predicted_token_proba_2, 
      label_per_token_logits)) = inference(model, rank, criterion, test_loader, train_label_frequency,
                                           test_file_ids=test_file_ids, pad_token_id=pad_token_id)

    prediction_result = pd.DataFrame({'id':test_file_ids,'predicted_probas':predicted_probas, 'labels':labels_})
    prediction_result.to_csv(os.path.join(save_path, f'inference_logit_{file_name}_{sentence_ps}_{window_size}.csv'), index=False)
    result_df = calc_metric(predicted_probas, labels_)
    result_df.to_csv(os.path.join(save_path, f'inference_thresholding_{file_name}_{sentence_ps}_{window_size}.csv'), index=False)

    file_result = pd.DataFrame({'id':file_ids_list, 'predicted_token_proba_0':predicted_token_proba_0,
                                'predicted_token_proba_1':predicted_token_proba_1, 
                                'predicted_token_proba_2':predicted_token_proba_2,
                                'label':label_per_token_logits})
    file_result.to_csv(os.path.join(save_path, f'inference_file_{file_name}_{sentence_ps}_{window_size}.csv'), index=False)
    print()

def parse_args():
    parser = argparse.ArgumentParser(description='Inference : GPT model : CE_Loss')
    parser.add_argument('--SEED', type=int, default=17, help='Random Seed')
    parser.add_argument("--sentence_ps", type=str, default='None')
    parser.add_argument("--window_size", type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
"""
CUDA_VISIBLE_DEVICES=2 python inference.py --sentence_ps None --window_size 3
CUDA_VISIBLE_DEVICES=3 python inference.py --sentence_ps moving_average --window_size 3

"""

