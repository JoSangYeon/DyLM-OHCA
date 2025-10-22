import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

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

from utils import DataParallelModel, DataParallelCriterion
from utils import set_device, calc_acc
from model import *
from dataset import *
from learning import evaluate

import warnings
warnings.filterwarnings(action='ignore')


SEED = 42
# random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_metric_result(predicted_probas, predicted_labels, true_labels):
    TH_ACC = accuracy_score(true_labels, predicted_labels)
    AUROC = roc_auc_score(true_labels, predicted_probas)
    AUPRC = average_precision_score(true_labels, predicted_probas)
    RECALL = recall_score(true_labels, predicted_labels)
    PRECISION = precision_score(true_labels, predicted_labels)
    F1 = f1_score(true_labels, predicted_labels)
    BRIER = brier_score_loss(true_labels, predicted_probas)
    # CM = confusion_matrix(labels, predicted_labels)
    return TH_ACC, AUROC, AUPRC, RECALL, PRECISION, F1, BRIER

def get_metric_df(metric_result):
    METRIC_INDEX = ['TH_ACC', 'AUROC', 'AUPRC', 'RECALL', 'PRECISION', 'F1', 'BRIER']
    df_dict = {'mean':[], 'std':[]}
    for i, mi in enumerate(METRIC_INDEX):
        metric_val = metric_result[:, i]
        mean_ = round(np.mean(metric_val), 4)
        std_ = round(np.std(metric_val), 3)
        
        df_dict['mean'].append(mean_)
        df_dict['std'].append(std_)
    return pd.DataFrame(df_dict, index=METRIC_INDEX)

# def inference(model,
#               main_device,
#               criterion,
#               label_frequency,
#               data_loader,):
#     METRIC = {
#         'file_name': [],
#         'max_seq_proba': [],
#         'mean_seq_proba': [],
#         'max_seq_idx': [],
#         'loss': [],
#         'acc' : [],
#         'LABEL': [],
#     }
#     label_tag = {0:'비응급', 1:'응급'}
    
#     model.eval()
    
#     max_predicted_probas = []
#     mean_predicted_probas = []
#     true_labels = []
#     with torch.no_grad():
#         pbar = tqdm(data_loader, file=sys.stdout)
#         for batch_idx, (file_path, samples) in enumerate(pbar):
#             # sample_file = data_loader.dataset.files[batch_idx]
#             sample_file = file_path[0]

#             input_ids, att_mask, type_ids, target = samples
#             mb_len = len(target)

#             input_ids = torch.stack(input_ids).view(mb_len, -1).to(main_device)
#             att_mask = torch.stack(att_mask).view(mb_len, -1).to(main_device)
#             type_ids = torch.stack(type_ids).view(mb_len, -1).to(main_device)
#             target = torch.stack(target).view(mb_len).to(main_device)

#             output = model(input_ids, att_mask, type_ids)
#             # output = get_each_output(output)
#             loss = criterion(output, target)
#             acc = calc_acc(output, target) / mb_len
            
#             logits = output.detach().cpu()
#             labels = target.detach().cpu()
            
#             predicted_probas = torch.softmax(logits, dim=-1)[:, 1] # (N, 2) -> (N, 1) -> (N, )
#             max_seq_proba, max_seq_idx = torch.max(predicted_probas, dim=0) # max_proba, max_idx
#             max_seq_proba, max_seq_idx = max_seq_proba.item(), max_seq_idx.item()
#             mean_seq_proba = torch.mean(predicted_probas).item()
#             label = labels[-1].item()
            
#             METRIC['file_name'].append(sample_file)
#             METRIC['max_seq_proba'].append(max_seq_proba)
#             METRIC['mean_seq_proba'].append(mean_seq_proba)
#             METRIC['max_seq_idx'].append(max_seq_idx)
#             METRIC['loss'].append(loss.item())
#             METRIC['acc'].append(acc)
#             METRIC['LABEL'].append(label_tag[label])
            
#             max_predicted_probas.append(max_seq_proba)
#             mean_predicted_probas.append(mean_seq_proba)
#             true_labels.append(label)
#             pbar.set_postfix(loss='{:.4f}, acc={:.4f}'.format(loss.item(), acc))
#         pbar.close()
        
#         max_predicted_probas = np.array(max_predicted_probas)
#         max_predicted_labels = np.where(max_predicted_probas >= label_frequency, 1, 0)
#         mean_predicted_probas = np.array(mean_predicted_probas)
#         mean_predicted_labels = np.where(mean_predicted_probas >= label_frequency, 1, 0)
#         true_labels = np.array(true_labels)
        
#         MAX_METRIC = get_metric_result(max_predicted_probas, max_predicted_labels, true_labels)
#         MEAN_METRIC = get_metric_result(mean_predicted_probas, mean_predicted_labels, true_labels)        
#     return METRIC, MAX_METRIC, MEAN_METRIC


def calc_th_acc(output, target, label_frequency):
    logits = output.detach().cpu()
    labels = target.detach().cpu()
    
    predicted_probas = torch.softmax(logits, dim=-1)[:, 1] # (N, 2) -> (N, 1)
    max_predicted_labels = torch.where(predicted_probas >= label_frequency, 1, 0)
    
    return (max_predicted_labels == labels).sum().item()

def inference(model,
              main_device,
              criterion,
              label_frequency,
              data_loader,
              sma_window_size=3,):
    METRIC = {
        'file_name': [],
        'max_seq_proba': [],
        'max_seq_idx': [],
        'mean_seq_proba': [],
        'sma_mean_seq_proba': [],
        'sma_idx': [],
        'loss': [],
        'acc' : [],
        'LABEL': [],
    }
    label_tag = {0:'비응급', 1:'응급'}
    
    model.eval()
    tokenizer = data_loader.dataset.tokenizer
    
    max_predicted_probas = []
    mean_predicted_probas = []
    sma_mean_seq_probas = []
    true_labels = []
    with torch.no_grad():
        pbar = tqdm(data_loader, file=sys.stdout)
        for batch_idx, (file_path, samples) in enumerate(pbar):
            # sample_file = data_loader.dataset.files[batch_idx]
            sample_file = file_path[0]

            input_ids, att_mask, type_ids, target = samples
            mb_len = len(target)

            input_ids = torch.stack(input_ids).view(mb_len, -1).to(main_device)
            att_mask = torch.stack(att_mask).view(mb_len, -1).to(main_device)
            type_ids = torch.stack(type_ids).view(mb_len, -1).to(main_device)
            target = torch.stack(target).view(mb_len).to(main_device)

            output = model(input_ids, att_mask, type_ids)
            # output = get_each_output(output)
            loss = criterion(output, target)
            acc = calc_th_acc(output, target, label_frequency) / mb_len
            
            logits = output.detach().cpu()
            labels = target.detach().cpu()
            
            predicted_probas = torch.softmax(logits, dim=-1)[:, 1] # (N, 2) -> (N, 1)
            max_seq_proba, max_seq_idx = torch.max(predicted_probas, dim=0) # max_proba, max_idx
            max_seq_proba, max_seq_idx = max_seq_proba.item(), max_seq_idx.item()
            mean_seq_proba = torch.mean(predicted_probas).item()
            if len(predicted_probas) < sma_window_size:
                sma_mean_seq_proba, sma_idx=torch.max(predicted_probas.unfold(0, len(predicted_probas), 1).mean(dim=-1), dim=0) #unfold(dim, window_size, step)
            else:
                sma_mean_seq_proba, sma_idx=torch.max(predicted_probas.unfold(0, sma_window_size, 1).mean(dim=-1), dim=0) #unfold(dim, window_size, step)
            sma_mean_seq_proba, sma_idx = sma_mean_seq_proba.item(), sma_idx.item()
            label = labels[-1].item()
            
            METRIC['file_name'].append(sample_file)
            METRIC['max_seq_proba'].append(max_seq_proba)
            METRIC['mean_seq_proba'].append(mean_seq_proba)
            METRIC['max_seq_idx'].append(max_seq_idx)
            METRIC['loss'].append(loss.item())
            METRIC['acc'].append(acc)
            METRIC['LABEL'].append(label_tag[label])

            METRIC['sma_mean_seq_proba'].append(sma_mean_seq_proba)
            METRIC['sma_idx'].append(sma_idx)
            
            max_predicted_probas.append(max_seq_proba)
            mean_predicted_probas.append(mean_seq_proba)
            sma_mean_seq_probas.append(sma_mean_seq_proba)
            true_labels.append(label)
            pbar.set_postfix(loss='{:.4f}, acc={:.4f}'.format(loss.item(), acc))
            
            if acc < 0.5:
                wrong_sample = {'window': [], 'predicted_proba':[], 'label': []}
                for ids, proba in zip(input_ids, predicted_probas):
                    text = tokenizer.decode(ids)
                    text = text[5:text.find('[SEP]')]
                    
                    wrong_sample['window'].append(text)
                    wrong_sample['predicted_proba'].append(proba.item())
                    wrong_sample['label'].append(label)
                wrong_sample_df = pd.DataFrame(wrong_sample)
                wrong_sample_df.to_csv(os.path.join('wrong_samples', f'{os.path.basename(sample_file)}.csv'), index=False)
                
        pbar.close()
        
        max_predicted_probas = np.array(max_predicted_probas)
        max_predicted_labels = np.where(max_predicted_probas >= label_frequency, 1, 0)
        mean_predicted_probas = np.array(mean_predicted_probas)
        mean_predicted_labels = np.where(mean_predicted_probas >= label_frequency, 1, 0)
        sma_mean_seq_probas = np.array(sma_mean_seq_probas)
        sma_mean_predicted_labels = np.where(sma_mean_seq_probas >= label_frequency, 1, 0)

        true_labels = np.array(true_labels)
        
        MAX_METRIC = get_metric_result(max_predicted_probas, max_predicted_labels, true_labels)
        MEAN_METRIC = get_metric_result(mean_predicted_probas, mean_predicted_labels, true_labels)        
        SMA_METRIC = get_metric_result(sma_mean_seq_probas, sma_mean_predicted_labels, true_labels)        
    return METRIC, MAX_METRIC, MEAN_METRIC, SMA_METRIC

def test_model(model, ckpt_path, main_device, 
               test_data, tokenizer, label_frequency,
               sample_rate=0.49, bootstrap_K=10,):
    ckpt_root = os.path.dirname(ckpt_path)
    file_name = os.path.basename(ckpt_path).split('.')[0]
    ckpt = torch.load(ckpt_path, map_location=main_device)
    model.load_state_dict(ckpt['model_state_dict']); model.to(main_device)
    
    # # bootstrap으로 성능 확인
    # max_result, mean_result = [], []
    # for k in range(bootstrap_K):
    #     while True:
    #         file_list = test_data.id.unique()
    #         selected_file_list = np.random.choice(file_list, size=int(len(file_list)*sample_rate), replace=False)
    #         bootstrap_data = test_data[test_data.id.isin(selected_file_list)]
    #         bootstrap_data = bootstrap_data.reset_index(drop=True)
    #         # print(bootstrap_data.label.sum(), bootstrap_data.label.sum() > 1)
    #         if bootstrap_data.label.sum() > 1:
    #             break
        
    #     bootstrap_dataset = Sample_Metric_Dataset(bootstrap_data, tokenizer, 
    #                                               max_length=256, padding='max_length', class_num=2)
    #     data_loader = DataLoader(bootstrap_dataset, 
    #                              batch_size=1, 
    #                              sampler=RandomSampler(bootstrap_dataset))
    #     _, MAX_METRIC, MEAN_METRIC= inference(model,
    #                                           main_device,
    #                                           nn.CrossEntropyLoss(),
    #                                           label_frequency,
    #                                           data_loader)
        
    #     max_result.append(MAX_METRIC); mean_result.append(MEAN_METRIC)
    # max_result = np.array(max_result); mean_result = np.array(mean_result)
    
    # max_result_df = get_metric_df(max_result)
    # mean_result_df = get_metric_df(mean_result)    
    
    # 전체 test데이터로 가장 응급한 순간 체크
    test_dataset = Sample_Metric_Dataset(test_data, tokenizer,
                                         max_length=256,
                                         padding='max_length',
                                         class_num=2, )
    data_loader = DataLoader(test_dataset, batch_size=1, sampler=RandomSampler(test_dataset))
    
    METRIC, _, _, _ = inference(model,
                            main_device,
                            nn.CrossEntropyLoss(),
                            label_frequency,
                            data_loader)
        
    result_df = pd.DataFrame(METRIC, index=METRIC['file_name'])
    result_df.to_csv(os.path.join(ckpt_root, f'METRIC_{file_name}.csv'), index=False)
    
    # max_result_df.to_csv(os.path.join(ckpt_root, f'MAX_{file_name}.csv'))
    # mean_result_df.to_csv(os.path.join(ckpt_root, f'MEAN_{file_name}.csv'))
    
    return result_df

def display_sample_metric(metric: pd.DataFrame, k=10):
    df_1 = metric[metric.LABEL == '응급']

    df_1 = df_1.sample(k, random_state=SEED)
    for (file_name, max_seq_proba, mean_seq_proba, max_seq_idx, loss, acc, LABEL) in df_1.values:
        print(f'FILE : {file_name}: ')
        with open(file_name, 'r') as f:
            data = json.load(f)

            utterances = data['utterances']
            for id, content in enumerate(utterances):
                if id == max_seq_idx+5:
                    print(f"\t{id}. {content['text'].strip()}\n\t\t└---응급시점: {max_seq_proba:.6f}")
                else:
                    print(f"\t{id}. {content['text'].strip()}")
            print('\n')
            print("\tSample loss : {:.6f}".format(loss))
            print("\tSample acc : {:.3f}".format(acc))
        print('\n\n')


def main():
    # Define project
    project_name = 'NIA_119'
    model_name = 'Baseline_1'
    model_link = 'beomi/KcELECTRA-base-v2022' #'beomi/kcbert-base'
    
    # args
    epochs = 5
    batch_size = 64
    electra_lr = 1e-5
    cls_lr = 1e-2
    
    class_num = 2
    max_length = 384
    padding = 'max_length'

    main_device, device_ids = set_device(main_device_num=2, using_device_num=0)
    ckpt_path = os.path.join('models', 'Baseline_1_e7_bs64')
    ckpt_num = 2

    # Datasets
    train_path = "train_data.csv"
    valid_path = "valid_data.csv"
    test_path = "test_data.csv"

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)

    ## your Data Pre-Processing
    print('init train data :', train_data.shape)
    print('init valid data :', valid_data.shape)
    print('init test data :', test_data.shape)

    train_data = train_data.dropna(axis=0)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.dropna(axis=0)
    valid_data = valid_data.reset_index(drop=True)
    test_data = test_data.dropna(axis=0)
    test_data = test_data.reset_index(drop=True)

    print('Drop nan(train) :', train_data.shape)
    print('Drop nan(valid) :', valid_data.shape)
    print('Drp nan(test) :', test_data.shape)
    
    ## Create Dataset and DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_link)
    
    ## data split
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
    test_dataset = Sample_Metric_Dataset(test_data,
                                         tokenizer,
                                         max_length=max_length,
                                         padding=padding,
                                         class_num=class_num)

    ## Check Train, Valid, Test Dataset Shape
    print("The Length of Train Data: ", len(train_dataset))
    print("The Length of Valid Data: ", len(valid_dataset))
    print("The Length of Test Data: ", len(test_dataset))

    # label_frequency
    label_frequency = train_dataset.data['label'].sum() / len(train_dataset) # baselin_0의 경우
    # label_frequency = 0.19431177110924364 # baseline_1의 경우
    # label_frequency = 0.105882 # baseline_0
    # label_frequency = 0.219608 # baseline_1
    print('THRESHOLD : {:6f}'.format(label_frequency))
    
    # Inference Best Model 
    init_model = Baseline(model_link=model_link, class_num=2)
    result = test_model(init_model, ckpt_path, ckpt_num, main_device,
                        test_data, tokenizer, label_frequency,
                        sample_rate=0.65, bootstrap_K=5,)
    
    # display_sample_metric(result, k=10)


if __name__ == '__main__':
    # test_path = "test_data.csv"
    # test_data = pd.read_csv(test_path)

    # print('init test data :', test_data.shape)
    
    # ckpt_num = 1
    # result = pd.read_csv(f'models/Baseline_1_e5_bs64/METRIC_ckpt_{ckpt_num}.csv')
    # display_sample_metric(result, k=20)
    main()