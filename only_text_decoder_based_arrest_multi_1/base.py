# https://github.com/HideOnHouse/TorchBase

import os
import glob
import wandb
import pickle
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler

from transformers import GPT2TokenizerFast
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from dataset import *
from learning import *
from model import *
# from data_processing import *
# from inference import *
# from inference_switching_recall import *
from utils import set_device, set_save_path, set_label_frequency, str2bool, calc_metric

import warnings
warnings.filterwarnings(action='ignore')


def main():
    args = parse_args()

    world_size = 1
    is_stacked = args.IS_STACKED
    is_norm = args.IS_NORM
    
    # Define project
    model_name = f'Only_Text_{"stacked" if is_stacked else "window"}_{"normO" if is_stacked else "normX"}'
    model_link = "skt/kogpt2-base-v2" #'beomi/KcELECTRA-base-v2022' #'beomi/kcbert-base'

    # args
    epochs = 100
    batch_size = 48
    lr = 1e-5
    electra_lr = 1e-5
    cls_lr = 1e-3
    
    class_num = 2
    speaker_num = 4
    max_length = 256
    padding = 'max_length'
    save_term = 1024 * 5
    
    # train_path = os.path.join('NIA_text_dataset', 'train_json_audio_data_end_stack_speaker.csv')
    # valid_path = os.path.join('NIA_text_dataset', 'valid_json_audio_data_end_stack_speaker.csv')
    # test_path = os.path.join('NIA_text_dataset', 'test_json_audio_data_end_stack_speaker.csv')
    train_path = os.path.join('.', 'decoder_based_data.csv')
    valid_path = os.path.join('.', 'decoder_based_data.csv')
    test_path = os.path.join('.', 'decoder_based_data.csv')
        
    save_path = set_save_path(model_name, epochs, batch_size)

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)
    valid_file_ids = valid_data.id
    test_file_ids = test_data.id

    ## your Data Pre-Processing
    # train_data = train_data.dropna(axis=0)
    train_data = train_data.reset_index(drop=True)
    # valid_data = valid_data.dropna(axis=0)
    valid_data = valid_data.reset_index(drop=True)
    # test_data = test_data.dropna(axis=0)
    test_data = test_data.reset_index(drop=True)

    print('train data :', train_data.shape)
    print('valid data :', valid_data.shape)
    print('test data :', test_data.shape)
    
    ## Create Dataset and DataLoader
    tokenizer = GPT2TokenizerFast.from_pretrained(model_link,bos_token='</s>', eos_token='</s>', 
                                                  unk_token='<unk>',pad_token='<pad>', mask_token='<mask>')
    special_tokens_dict = {'additional_special_tokens': [f'[SPK{n}]' for n in range(speaker_num)]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=world_size*4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=world_size*4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=world_size*4)

    ## label_frequency
    train_label_frequency = (train_data.label == 1).sum() / len(train_data)
    valid_label_frequency = (valid_data.label == 1).sum() / len(valid_data)
    test_label_frequency = (test_data.label == 1).sum() / len(test_data)

    print("Label frequency of Train Data: {:6f}".format(train_label_frequency))
    print("Label frequency of Valid Data: {:6f}".format(valid_label_frequency))
    print("Label frequency of Test Data: {:6f}".format(test_label_frequency))
    
    # modeling
    rank = 'cuda:0'
    model = GPT_Baseline.from_pretrained(model_link)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(rank)

    # optimizer = optim.AdamW([{'params': model.module.electra.parameters(),'lr': electra_lr},
    #                          {'params': model.module.classifier.parameters(),'lr': cls_lr}],
    #                         eps=1e-8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss()

    # config = {
    #     'total_lr': lr,
    #     'electra_lr': electra_lr,
    #     'cls_lr': cls_lr,
    #     'batch_size': batch_size,
    #     'epochs': epochs,
    #     'max_length': max_length,
    #     'train_label_frequency': train_label_frequency,
    #     'valid_label_frequency': valid_label_frequency,
    #     'test_label_frequency': test_label_frequency,
    # }
    # wandb.config.update(config)
    
    # Train & valid
    train_func = train_normed if is_norm else train
    eval_func = evaluate_normed if is_norm else evaluate
    print("============================= Train =============================")
    _ = train_func(rank, model, optimizer, criterion, epochs, save_path, 
                   train_loader, None, valid_loader, None, 
                   save_term, train_label_frequency)

    # if rank == 0: # Test
    #     print("============================= Test & Inference =============================")
    #     ckpt_path = sorted(glob.glob(os.path.join(save_path, '*.tar')), key=lambda x : (int(x[:-4].split('_')[-2]) * int(x[:-4].split('_')[-1])) % int(x[:-4].split('_')[-2]), reverse=True)
    #     # ckpt_path = sorted(glob.glob(os.path.join(save_path, '*.tar')))
    #     print(ckpt_path)
    #     # print(ckpt_path)
    #     for model_path in ckpt_path:
    #         print(f"{model_path} >>> ")
    #         file_name = os.path.basename(model_path).split('.')[0]
    #         model = Baseline.from_pretrained(model_link)
    #         model.resize_token_embeddings(len(tokenizer))
    #         ckpt = torch.load(model_path, map_location=f'cuda:{rank}')
    #         model.load_state_dict(ckpt['model_state_dict']); model.to(rank)
            
    #         # test_loss, test_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER), (predicted_probas, labels)= inference_evaluate(model, 
    #         #                                                                                                                           main_device, 
    #         #                                                                                                                           criterion, 
    #         #                                                                                                                           train_label_frequency, 
    #         #                                                                                                                           valid_loader)
    #         # print("test loss : {:.6f}".format(test_loss))
    #         # print("test acc : {:.3f}".format(test_acc))
    #         # print("test acc(th) : {:4f}".format(TH_ACC))
    #         # print("test AUROC : {:.4f}".format(AUROC))
    #         # print("test AUPRC : {:.4f}".format(AUPRC))
    #         # print("test Recall : {:4f}".format(RECALL))    
    #         # print("test Precision : {:.4f}".format(PRECISION))
    #         # print("test F1_score : {:.4f}".format(F1))
    #         # print("test Brier : {:4f}".format(BRIER))
    #         # prediction_values = pd.DataFrame({'id':valid_file_ids,'predicted_probas':predicted_probas, 'labels':labels})
    #         # prediction_values.to_csv(os.path.join(save_path, f'value_valid_{file_name}.csv'), index=False)
            
    #         (test_loss, test_acc, 
    #          (AUROC, AUPRC, TH_ACC, RECALL, 
    #           PRECISION, F1, BRIER), 
    #          (predicted_probas, labels)) = eval_func(model, rank,  criterion, 
    #                                                  test_loader, train_label_frequency, 
    #                                                  is_infernece=True)
    #         print("test loss : {:.6f}".format(test_loss))
    #         print("test acc : {:.3f}".format(test_acc))
    #         print("test acc(th) : {:4f}".format(TH_ACC))
    #         print("test AUROC : {:.4f}".format(AUROC))
    #         print("test AUPRC : {:.4f}".format(AUPRC))
    #         print("test Recall : {:4f}".format(RECALL))    
    #         print("test Precision : {:.4f}".format(PRECISION))
    #         print("test F1_score : {:.4f}".format(F1))
    #         print("test Brier : {:4f}".format(BRIER))
    #         prediction_values = pd.DataFrame({'id':test_file_ids,'predicted_probas':predicted_probas, 'labels':labels})
    #         prediction_values.to_csv(os.path.join(save_path, f'value_test_{file_name}.csv'), index=False)
    #         result_df = calc_metric(predicted_probas, labels)
    #         result_df.to_csv(os.path.join(save_path, f'TEST_{file_name}.csv'), index=False)
    #         print()

def parse_args():
    parser = argparse.ArgumentParser(description='Training : Stacked model')
    parser.add_argument('--SEED', type=int, default=17, help='Random Seed')
    parser.add_argument('--WORLD_SIZE', type=int, default=2, help='number of distributed processes')
    parser.add_argument('--PORT', type=str, default='12356', help='number of Master PORT Number')
    parser.add_argument("--IS_STACKED", type=str2bool, default=False)
    parser.add_argument("--IS_NORM", type=str2bool, default=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=3 python base.py
"""