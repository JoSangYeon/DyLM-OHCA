import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, notebook

import torch

from sklearn.metrics import (roc_auc_score, 
                             average_precision_score,
                             accuracy_score,  
                             recall_score,
                             precision_score,
                             f1_score,
                             brier_score_loss, # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
                             confusion_matrix,)

def calc_metric(predicted_probas, labels):
    result = {
        'THRESHOLD' : [],
        
        'TH_ACC' : [],
        'AUROC' : [],
        'AUPRC' : [],
        'RECALL' : [],
        'PRECISION' : [],
        'F1': [],
    }
    
    for threshold in np.linspace(0, 1, 384):
        result['THRESHOLD'].append(round(threshold, 5))
        
        predicted_label = np.where(predicted_probas >= threshold , 1, 0)
        
        # result['ACC'].append(round(accuracy_score(labels, np.where(predicted_probas >= 0.5 , 1, 0)), 5))
        result['TH_ACC'].append(round(accuracy_score(labels, predicted_label), 5))
        result['AUROC'].append(round(roc_auc_score(labels, predicted_probas), 5))
        result['AUPRC'].append(round(average_precision_score(labels, predicted_probas), 5))
        result['RECALL'].append(round(recall_score(labels, predicted_label), 5))
        result['PRECISION'].append(round(precision_score(labels, predicted_label), 5))
        result['F1'].append(round(f1_score(labels, predicted_label), 5))
        
    result_df = pd.DataFrame(result)
    return result_df

def set_last_sequence(df, end_time=120000, cut=False):
    columns = ['id', 'json_file_path', 'wav_original_file_path',
               'text', 'endAt', 'label']
    result = {
        'id' : [], 
        'json_file_path' : [], 
        'wav_original_file_path' : [],
        'text' : [], 
        'endAt' : [], 
        'label' : []
    }
    id_list = df.id.unique()
    if cut:        
        for i in tqdm(range(len(id_list))):
            temp = df[df.id == id_list[i]]
            temp = temp[temp.endAt <= end_time].iloc[-1:, :]
            id, json_file_path, wav_original_file_path, text, endAt, label = temp.values[0]
            result['id'].append(id)
            result['json_file_path'].append(json_file_path)
            result['wav_original_file_path'].append(wav_original_file_path)
            result['text'].append(text)
            result['endAt'].append(endAt)
            result['label'].append(label)
    else:   
        for i in tqdm(range(len(id_list))):
            temp = df[df.id == id_list[i]].iloc[-1:, :]
            id, json_file_path, wav_original_file_path, text, endAt, label = temp.values[0]
            result['id'].append(id)
            result['json_file_path'].append(json_file_path)
            result['wav_original_file_path'].append(wav_original_file_path)
            result['text'].append(text)
            result['endAt'].append(endAt)
            result['label'].append(label)
    
    one_seq_df = pd.DataFrame(result)
    return one_seq_df.sample(frac=1).reset_index(drop=True)

def set_label_frequency(df, rate=0.35, target_label=1, by_file=True):
    if rate == 0:
        label_frequency = (df.label == 1).sum() / len(df)
        return df, label_frequency
    
    columns = ['file_path', 'id', 'text', 'endAt', 'label']
    if by_file:
        upsampling_file_list = df[df.label == target_label].id.unique()
        np.random.shuffle(upsampling_file_list)
        N = int(len(upsampling_file_list) * rate)
        file_list = upsampling_file_list[:N]
        upsampling_df = df[df.id.isin(file_list)]
        target_label_df = df[df.label == target_label]
        non_target_label_df = df[df.label != target_label]
    else:
        upsampling_df = df[df.label == target_label].sample(frac=rate)
        target_label_df = df[df.label == target_label]
        non_target_label_df = df[df.label != target_label]
    result_df = pd.concat([target_label_df, non_target_label_df, upsampling_df], axis=0).sample(frac=1).reset_index(drop=True)
    label_frequency = (result_df.label == 1).sum() / len(result_df)
    return result_df, label_frequency


def get_each_output(output):
    if type(output) is not list:
        return output
    else:
        return list(map(list, zip(*output)))

def calc_acc(output, label):
    if type(output) is list:  # if multi-gpu settings
        # copy to cpu from gpu
        output = [o.detach().cpu() for o in output]
        output = torch.cat(output, dim=0)
        label = label.detach().cpu()

    o_val, o_idx = torch.max(output, dim=-1)
    return (o_idx == label).sum().item()


def draw_history(history, save_path=None):
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    valid_loss = history["valid_loss"]
    valid_acc = history["valid_acc"]

    plt.subplot(2, 1, 1)
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="valid")
    plt.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_path, 'train_plot.png'), dpi=300)


def set_device(main_device_num=0, using_device_num=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_ids = list(range(main_device_num, main_device_num + using_device_num))
    if device == 'cuda':
        device += ':{}'.format(main_device_num)
    return device, device_ids


def set_save_path(model_name, epochs, batch_size):
    directory = os.path.join('models', f'{model_name}_e{epochs}_bs{batch_size}')
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_name_ext(file_path: str) -> tuple[str, str]:
    """
    :param file_path: absolute or relative file path where file is located
    :return: name, extension
    """
    if os.sep in file_path:
        file_name = file_path.split(os.sep)[-1]
    else:
        file_name = file_path
    if os.extsep in file_name:
        name, ext = file_name.rsplit(os.extsep, maxsplit=1)
    else:
        name, ext = file_name, ""
    return name,
