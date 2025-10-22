import os
import sys
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, notebook
from einops import rearrange, repeat

from sklearn.metrics import (roc_auc_score, 
                             average_precision_score,
                             accuracy_score,  
                             recall_score,
                             precision_score,
                             f1_score,
                             brier_score_loss, # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
                             confusion_matrix,)
from imblearn.metrics import specificity_score

from utils import get_each_output, calc_acc

def get_loss(rank, input_ids, logits, labels, criterion, pad_token_id=0):
    labelss = labels[:, -1] - 1 # batch

    loss_sum = torch.tensor(0.).to(rank)
    batch_size = input_ids.size(0)
    
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    sequence_lengths = sequence_lengths.to(rank)
    
    for i, seq_len in enumerate(sequence_lengths):
        logit = logits[i, :seq_len]
        label = repeat(labelss[i], ' -> n', n=seq_len).long()
        loss_sum += criterion(logit, label)
    return loss_sum / batch_size

def train(rank, model, optimizer, criterion, epochs, save_path, train_loader=None, train_sampler=None,
          valid_loader=None, valid_sampler=None, save_term=2048, label_frequency=0.5, pad_token_id=0):
    model.to(rank)
    bs = train_loader.batch_size
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch) if train_sampler is not None else ''
        
        ctc_loss_lst = []
        ce_loss_lst = []
        acc_lst = []
        
        if rank == 0 or rank =='cuda:0':
            train_pbar = tqdm(train_loader, file=sys.stdout)
        else:
            train_pbar = train_loader
            
        for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), (target, target_length)) in enumerate(train_pbar):
            input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)
            type_ids, spk_type_ids = type_ids.to(rank), type_ids.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            optimizer.zero_grad()
            output, logit, ctc_loss = model(input_ids=input_ids, attention_mask=att_mask,
                                            token_type_ids=type_ids, speaker_type_ids=spk_type_ids,
                                            target=target, target_length=target_length)
            
            # output = get_each_output(output)
            # loss = criterion(output, target)
            ce_loss = get_loss(rank, input_ids, output, target, nn.CrossEntropyLoss())
            # acc = get_acc(rank, input_ids, output, target, criterion)
            acc = calc_acc(logit, target[:, -1] - 1)
            ctc_loss.backward()
            optimizer.step()

            ctc_loss_lst.append(ctc_loss.item());  ce_loss_lst.append(ce_loss.item()); acc_lst.append(acc)
            if rank == 0 or rank =='cuda:0':
                train_pbar.set_postfix_str(f'epoch={epoch}/{epochs}, CTC={np.mean(ctc_loss_lst):.6f}, CE={np.mean(ce_loss_lst):.6f} acc={np.sum(acc_lst) / (batch_idx * bs + mb_len):.4f}')
            
            if (rank == 0 or rank =='cuda:0') and batch_idx != 0 and batch_idx % save_term == 0:
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'epoch': epoch,
                    'batch_idx': batch_idx
                }, os.path.join(save_path, f'checkpoint_{epoch}_{batch_idx}.tar'))

        if rank == 0 or rank =='cuda:0':
            train_pbar.close()
            wandb.log({
                'train_ctc_loss' : np.mean(ctc_loss_lst),
                'train_ce_loss' : np.mean(ce_loss_lst),
                'train_acc' : np.sum(acc_lst) / (batch_idx * bs + mb_len)
            })

        if valid_loader is not None:
            valid_sampler.set_epoch(epoch) if valid_sampler is not None else ''
            (valid_ctc_loss, valid_ce_loss, valid_acc, 
            (AUROC, AUPRC, TH_ACC, RECALL, 
            PRECISION, SPECIFICITY, F1, BRIER)) = evaluate(model, rank, criterion, 
                                                           valid_loader, label_frequency)
            if rank == 0 or rank =='cuda:0':
                print("valid ctc loss : {:.6f}".format(valid_ctc_loss))
                print("valid ctc loss : {:.6f}".format(valid_ce_loss))
                print("valid acc : {:.3f}".format(valid_acc))
                print("valid acc(th) : {:4f}".format(TH_ACC))
                print("valid AUROC : {:.4f}".format(AUROC))
                print("valid AUPRC : {:.4f}".format(AUPRC))
                print("valid Recall : {:4f}".format(RECALL))    
                print("valid Precision : {:.4f}".format(PRECISION))
                print("valid_Specificity : {:.4f}".format(SPECIFICITY))
                print("valid F1_score : {:.4f}".format(F1))
                print("valid Brier : {:4f}".format(BRIER))
                print()
                wandb.log({
                'valid_ctc_loss' : valid_ctc_loss,
                'valid_ce_loss' : valid_ce_loss,
                'valid_acc' : valid_acc,
                'valid_acc(th)' : TH_ACC,
                'valid_AUROC' : AUROC,
                'valid_AUPRC' : AUPRC,
                'valid_Recall' : RECALL,
                'valid_Precision' : PRECISION,
                'valid_Specificity' : SPECIFICITY,
                'valid_F1_score' : F1,
                'valid_Brier' : BRIER,
                })

        if rank == 0 or rank =='cuda:0':
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'epoch': epoch,
                'batch_idx': batch_idx
            }, os.path.join(save_path, f'checkpoint_{epoch}_{batch_idx}.tar'))
    return model

def evaluate(model, rank, criterion, data_loader, label_frequency, is_inference=False, pad_token_id=0):
    model.eval()
    ctc_loss_lst = []
    ce_loss_lst = []
    acc_lst = []
    bs = data_loader.batch_size
    
    predicted = torch.tensor([])
    labels = torch.tensor([])

    with torch.no_grad():
        
        if rank == 0 or rank =='cuda:0':
            pbar = tqdm(data_loader, file=sys.stdout)
        else:
            pbar = data_loader
            
        for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), (target, target_length)) in enumerate(pbar):
            input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)
            type_ids, spk_type_ids = type_ids.to(rank), type_ids.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            output, logit, ctc_loss = model(input_ids=input_ids, attention_mask=att_mask,
                                            token_type_ids=type_ids, speaker_type_ids=spk_type_ids,
                                            target=target, target_length=target_length)

            ce_loss = get_loss(rank, input_ids, output, target, nn.CrossEntropyLoss())
            acc = calc_acc(logit, target[:, -1] - 1)

            ctc_loss_lst.append(ctc_loss.item());  ce_loss_lst.append(ce_loss.item()); acc_lst.append(acc)
            
            if rank == 0 or rank =='cuda:0':
                pbar.set_postfix_str(f'CTC={np.mean(ctc_loss_lst):.6f}, CE={np.mean(ce_loss_lst):.6f} acc={np.sum(acc_lst) / (batch_idx * bs + mb_len):.4f}')
            
            output_pred = logit.detach().cpu()
            true_label = (target[:, -1] - 1).detach().cpu()
            predicted = torch.concat([predicted, output_pred], dim=0)
            labels = torch.concat([labels, true_label], dim=0)
        if rank == 0 or rank =='cuda:0':
            pbar.close()

    total_ctc_loss = np.mean(ctc_loss_lst)
    total_ce_loss = np.mean(ce_loss_lst)
    total_acc = np.sum(acc_lst) / (batch_idx * bs + mb_len)
    
    # predicted_probas = torch.sigmoid(predicted)[:, 1]
    predicted_probas = predicted[:, 1]
    predicted_labels = torch.where(predicted_probas >= label_frequency , 1, 0)
    
    predicted_probas = predicted_probas.numpy()
    predicted_labels = predicted_labels.numpy()
    labels = labels.numpy()
    
    AUROC = roc_auc_score(labels, predicted_probas)
    AUPRC = average_precision_score(labels, predicted_probas)
    TH_ACC = accuracy_score(labels, predicted_labels)
    RECALL = recall_score(labels, predicted_labels)
    PRECISION = precision_score(labels, predicted_labels)
    SPECIFICITY = specificity_score(labels, predicted_labels)
    F1 = f1_score(labels, predicted_labels)
    BRIER = brier_score_loss(labels, predicted_probas)
    # CM = confusion_matrix(labels, predicted_labels)
    if is_inference:
        return total_ctc_loss, total_ce_loss, total_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, SPECIFICITY, F1, BRIER), (predicted_probas, labels)
    else:
        return total_ctc_loss, total_ce_loss, total_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, SPECIFICITY, F1, BRIER)

def inference(model, rank, criterion, data_loader, label_frequency, test_file_ids, pad_token_id=0):
    assert rank == 0

    model.eval()
    ctc_loss_lst = []
    ce_loss_lst = []
    acc_lst = []
    bs = data_loader.batch_size
    
    predicted = torch.tensor([])
    labels = torch.tensor([])

    file_ids_list = []
    input_ids_list = torch.tensor([])
    spk_type_ids_list = torch.tensor([])
    predicted_token_logits = torch.tensor([])
    label_per_token_logits = []

    with torch.no_grad():
        pbar = tqdm(data_loader, file=sys.stdout)
        for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), (target, target_length)) in enumerate(pbar):
            input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)
            type_ids, spk_type_ids = type_ids.to(rank), type_ids.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            output, logit, ctc_loss = model(input_ids=input_ids, attention_mask=att_mask,
                                            token_type_ids=type_ids, speaker_type_ids=spk_type_ids,
                                            target=target, target_length=target_length)
            
            sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(rank)

            ce_loss = get_loss(rank, input_ids, output, target, nn.CrossEntropyLoss())
            acc = calc_acc(logit, target[:, -1] - 1)

            ctc_loss_lst.append(ctc_loss.item());  ce_loss_lst.append(ce_loss.item()); acc_lst.append(acc)
            pbar.set_postfix_str(f'CTC={np.mean(ctc_loss_lst):.6f}, CE={np.mean(ce_loss_lst):.6f} acc={np.sum(acc_lst) / (batch_idx * bs + mb_len):.4f}')
            
            output_pred = logit.detach().cpu()
            true_label = (target[:, -1] - 1).detach().cpu()
            predicted = torch.concat([predicted, output_pred], dim=0)
            labels = torch.concat([labels, true_label], dim=0)

            input_ids = input_ids.detach().cpu()
            spk_type_ids = spk_type_ids.detach().cpu()
            for idx, (file_id, seq_len) in enumerate(zip(test_file_ids, sequence_lengths)):
                file_ids = [file_id] * seq_len; file_ids_list += file_ids
                input_id = input_ids[idx][:seq_len]; input_ids_list = torch.concat([input_ids_list, input_id], dim=0)
                spk_type_id = spk_type_ids[idx][:seq_len]; spk_type_ids_list = torch.concat([spk_type_ids_list, spk_type_id], dim=0)
                
                logits = output[idx, :seq_len]; logits=logits.detach().cpu()
                predicted_token_logits = torch.concat([predicted_token_logits, logits], dim=0)
                
                label = [true_label[idx].item()] * seq_len; label_per_token_logits += label
            test_file_ids = test_file_ids[bs:]

        pbar.close()

    total_ctc_loss = np.mean(ctc_loss_lst)
    total_ce_loss = np.mean(ce_loss_lst)
    total_acc = np.sum(acc_lst) / (batch_idx * bs + mb_len)
    
    # predicted_probas = torch.sigmoid(predicted)[:, 1]
    label_1_predicted_probas = predicted[:, 1]
    label_0_predicted_probas = predicted[:, 0]
    predicted_labels = torch.where(label_1_predicted_probas >= label_frequency , 1, 0)
    
    label_1_predicted_probas = label_1_predicted_probas.numpy()
    label_0_predicted_probas = label_0_predicted_probas.numpy()
    predicted_labels = predicted_labels.numpy()
    labels = labels.numpy()

    file_ids_list = np.array(file_ids_list)
    input_ids_list = input_ids_list.numpy().astype(np.int32)
    spk_type_ids_list = spk_type_ids_list.numpy().astype(np.int32)
    label_1_predicted_token_logits = predicted_token_logits[:, 1]
    label_0_predicted_token_logits = predicted_token_logits[:, 0]
    # predicted_token_logits = torch.where(predicted_token_logits >= label_frequency , 1, 0)
    predicted_token_logits = np.round(predicted_token_logits.numpy(), 6)
    label_per_token_logits = np.array(label_per_token_logits).astype(np.int32)

    return (label_1_predicted_probas, label_0_predicted_probas, labels), (file_ids_list, input_ids_list, spk_type_ids_list, label_1_predicted_token_logits, label_0_predicted_token_logits, label_per_token_logits)


def main():
    pass


if __name__ == "__main__":
    main()
