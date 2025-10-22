# https://github.com/HideOnHouse/TorchBase

"""
# gpt3-kor-small_based_on_gpt2
from transformers import BertTokenizerFast, GPT2Model
tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
model = GPT2Model.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
"""


import os
import glob
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

from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
from transformers import BertTokenizerFast
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from dataset import *
from learning import *
from model import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

def setup(rank, world_size, port, seed):
    # random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, port, seed, args):
    setup(rank, world_size, port, seed)
    
    # Define project
    project_name = f'NIA_119-GPT_응급_new'
    model_name = 'GPT-based_new_1'
    model_link = "kykim/gpt3-kor-small_based_on_gpt2" # 'skt/kogpt2-base-v2' #'beomi/kcbert-base'

    if rank==0:
        WANDB_AUTH_KEY = 'df1bca81589e9de3f6b797bf9af026b4d175e284'
        wandb.login(key=WANDB_AUTH_KEY)
        wandb.init(project=project_name)
        wandb.run.name = model_name

    # args
    epochs = 10
    batch_size = 12
    lr = 1e-5
    
    class_num = 2
    speaker_num = 4
    max_length = 768
    padding = 'max_length'
    save_term = 710
    
    # dataset
    train_path = os.path.join('..', 'NIA_text_dataset', 'train_json_audio_data_decoder_time_basic.csv')
    valid_path = os.path.join('..', 'NIA_text_dataset', 'valid_json_audio_data_decoder_time_basic.csv')
    test_path = os.path.join('..', 'NIA_text_dataset', 'test_json_audio_data_decoder_time_basic.csv')
    # train_path = os.path.join('..', 'NIA_text_dataset', 'toy_data_json_audio_data_decoder_time_basic.csv')
    # valid_path = os.path.join('..', 'NIA_text_dataset', 'toy_data_json_audio_data_decoder_time_basic.csv')
    # test_path = os.path.join('..', 'NIA_text_dataset', 'toy_data_json_audio_data_decoder_time_basic.csv')
        
    save_path = set_save_path(model_name, epochs, batch_size)

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)
    valid_file_ids = valid_data.id
    test_file_ids = test_data.id

    ## your Data Pre-Processing
    if rank == 0:
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
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4*world_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4*world_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4*world_size)

    ## label_frequency
    train_label_frequency = (train_data.label == 1).sum() / len(train_data)
    valid_label_frequency = (valid_data.label == 1).sum() / len(valid_data)
    test_label_frequency = (test_data.label == 1).sum() / len(test_data)
    if rank == 0:
        print("Label frequency of Train Data: {:6f}".format(train_label_frequency))
        print("Label frequency of Valid Data: {:6f}".format(valid_label_frequency))
        print("Label frequency of Test Data: {:6f}".format(test_label_frequency))
    
    # modeling
    model = GPT_Baseline.from_pretrained(model_link, class_num=class_num,
                                         pad_token_id=pad_token_id, cls_token_id=cls_token_id, sep_token_id=sep_token_id)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
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

    config = {
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'max_length': max_length,
        'train_label_frequency': train_label_frequency,
        'valid_label_frequency': valid_label_frequency,
        'test_label_frequency': test_label_frequency,
    }
    if rank==0: wandb.config.update(config)
    
    # # Train
    # print("============================= Train =============================")
    # _ = train(train_label_frequency, scheduler, model, main_device, optimizer, criterion, 
    #           epochs, save_path, train_loader, valid_loader, save_term)
    
    # Train
    if rank == 0: print("============================= Train =============================")
    _ = train(rank, ddp_model, optimizer, criterion, epochs, save_path,
              train_loader, train_sampler, valid_loader, valid_sampler, 
              save_term, train_label_frequency)

    # Test
    if rank==0:
        print("============================= Test & Inference =============================")
        ckpt_path = sorted(glob.glob(os.path.join(save_path, '*.tar')), key=lambda x : (int(x[:-4].split('_')[-2]) * int(x[:-4].split('_')[-1])) % int(x[:-4].split('_')[-2]), reverse=True)
        print(ckpt_path)
        # print(ckpt_path)
        for model_path in ckpt_path:
            print(f"{model_path} >>> ")
            file_name = os.path.basename(model_path).split('.')[0]
            model = GPT_Baseline.from_pretrained(model_link, class_num=class_num, pad_token_id=pad_token_id, 
                                                 cls_token_id=cls_token_id, sep_token_id=sep_token_id)
            model.resize_token_embeddings(len(tokenizer))
            ckpt = torch.load(model_path, map_location=f'cuda:{rank}')
            model.load_state_dict(ckpt['model_state_dict']); model.to(rank)
            
            # test_loss, test_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER), (predicted_probas, labels)= inference_evaluate(model, 
            #                                                                                                                           main_device, 
            #                                                                                                                           criterion, 
            #                                                                                                                           train_label_frequency, 
            #                                                                                                                           valid_loader)
            # print("test loss : {:.6f}".format(test_loss))
            # print("test acc : {:.3f}".format(test_acc))
            # print("test acc(th) : {:4f}".format(TH_ACC))
            # print("test AUROC : {:.4f}".format(AUROC))
            # print("test AUPRC : {:.4f}".format(AUPRC))
            # print("test Recall : {:4f}".format(RECALL))    
            # print("test Precision : {:.4f}".format(PRECISION))
            # print("test F1_score : {:.4f}".format(F1))
            # print("test Brier : {:4f}".format(BRIER))
            # prediction_values = pd.DataFrame({'id':valid_file_ids,'predicted_probas':predicted_probas, 'labels':labels})
            # prediction_values.to_csv(os.path.join(save_path, f'value_valid_{file_name}.csv'), index=False)
            
            ((predicted_probas, labels), 
             (file_ids_list, predicted_token_proba_0,
              predicted_token_proba_1, label_per_token_logits)) = inference(model, rank, criterion, test_loader, train_label_frequency,
                                                                            test_file_ids=test_file_ids, pad_token_id=pad_token_id)
            # input_tokens = []
            # for input_id in input_ids_list:
            #     input_tokens.append(tokenizer.decode(input_id))
            # print("test loss : {:.6f}".format(test_loss))
            # print("test acc : {:.3f}".format(test_acc))
            # print("test acc(th) : {:4f}".format(TH_ACC))
            # print("test AUROC : {:.4f}".format(AUROC))
            # print("test AUPRC : {:.4f}".format(AUPRC))
            # print("test Recall : {:4f}".format(RECALL))    
            # print("test Precision : {:.4f}".format(PRECISION))
            # print("valid_Specificity : {:.4f}".format(SPECIFICITY))
            # print("test F1_score : {:.4f}".format(F1))
            # print("test Brier : {:4f}".format(BRIER))
            prediction_result = pd.DataFrame({'id':test_file_ids,'predicted_probas':predicted_probas, 'labels':labels})
            prediction_result.to_csv(os.path.join(save_path, f'result_logit_{file_name}.csv'), index=False)
            result_df = calc_metric(predicted_probas, labels)
            result_df.to_csv(os.path.join(save_path, f'result_thresholding_{file_name}.csv'), index=False)

            file_result = pd.DataFrame({'id':file_ids_list, 'predicted_token_proba_0':predicted_token_proba_0,
                                        'predicted_token_proba_1':predicted_token_proba_1, 
                                        'label':label_per_token_logits})
            file_result.to_csv(os.path.join(save_path, f'result_file_{file_name}.csv'), index=False)
            print()

def parse_args():
    parser = argparse.ArgumentParser(description='Training : GPT model : CE_Loss')
    parser.add_argument('--SEED', type=int, default=17, help='Random Seed')
    parser.add_argument('--WORLD_SIZE', type=int, default=2, help='number of distributed processes')
    parser.add_argument('--PORT', type=str, default='12322', help='number of Master PORT Number')
    # parser.add_argument("--IS_AVGPOOL", type=str2bool, default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.spawn(main, args=(args.WORLD_SIZE, args.PORT, args.SEED, args), 
                                nprocs=args.WORLD_SIZE, join=True)
"""
CUDA_VISIBLE_DEVICES=1,2,3 python main.py --WORLD_SIZE 3

"""