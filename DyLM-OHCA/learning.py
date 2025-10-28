import os
import sys
import wandb
import numpy as np

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
    loss_sum = torch.tensor(0.).to(rank)
    batch_size = input_ids.size(0)
    
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    sequence_lengths = sequence_lengths.to(rank)
    
    for i, seq_len in enumerate(sequence_lengths):
        logit = logits[i, :seq_len]
        label = repeat(labels[i], ' -> n', n=seq_len).long()
        loss_sum += criterion(logit, label)
    return loss_sum / batch_size

def get_acc(rank, input_ids, logits, labels, criterion, pad_token_id=0):
    loss_sum = torch.tensor(0.).to(rank)
    batch_size = input_ids.size(0)
    
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    sequence_lengths = sequence_lengths.to(rank)
    
    logit = logits[torch.arange(batch_size, device=f'cuda:{rank}'), sequence_lengths]
    return calc_acc(logit, labels)

def train(rank, model, optimizer, criterion, epochs, save_path, train_loader=None, train_sampler=None,
          valid_loader=None, valid_sampler=None, save_term=2048, label_frequency=0.5, pad_token_id=0):
    model.to(rank)
    bs = train_loader.batch_size
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch) if train_sampler is not None else ''
        
        loss_lst = []
        acc_lst = []
        
        if rank == 0 or rank =='cuda:0':
            train_pbar = tqdm(train_loader, file=sys.stdout)
        else:
            train_pbar = train_loader
            
        for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), target) in enumerate(train_pbar):
            input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)
            type_ids, spk_type_ids = type_ids.to(rank), type_ids.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            optimizer.zero_grad()
            output, logit, seq_logits = model(input_ids=input_ids, attention_mask=att_mask,
                                              token_type_ids=type_ids, speaker_type_ids=spk_type_ids,
                                              is_inference=False)
            
            # output = get_each_output(output)
            # loss = criterion(output, target)
            loss = get_loss(rank, input_ids, output, target, criterion)
            # acc = get_acc(rank, input_ids, output, target, criterion)
            acc = calc_acc(logit, target)
            loss.backward()
            optimizer.step()

            loss_lst.append(loss.item()); acc_lst.append(acc)
            if rank == 0 or rank =='cuda:0':
                train_pbar.set_postfix(epoch=f'{epoch}/{epochs}', loss='{:.8f}, acc={:.4f}'.format(np.mean(loss_lst), np.sum(acc_lst) / (batch_idx * bs + mb_len)))
            
            if (rank == 0 or rank =='cuda:0') and batch_idx != 0 and batch_idx % save_term == 0:
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'epoch': epoch,
                    'batch_idx': batch_idx
                }, os.path.join(save_path, f'checkpoint_{epoch}_{batch_idx}.tar'))

        if rank == 0 or rank =='cuda:0':
            train_pbar.close()
            wandb.log({
                'train_loss' : np.mean(loss_lst),
                'train_acc' : np.sum(acc_lst) / (batch_idx * bs + mb_len)
            })

        if valid_loader is not None:
            valid_sampler.set_epoch(epoch) if valid_sampler is not None else ''
            (valid_loss, valid_acc, 
            (AUROC, AUPRC, TH_ACC, RECALL, 
            PRECISION, SPECIFICITY, F1, BRIER)) = evaluate(model, rank, criterion, 
                                                           valid_loader, label_frequency)
            if rank == 0 or rank =='cuda:0':
                print("valid loss : {:.6f}".format(valid_loss))
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
                'valid_loss' : valid_loss,
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
    sum_loss = sum_acc = 0
    bs = data_loader.batch_size
    
    predicted = torch.tensor([])
    labels = torch.tensor([])

    with torch.no_grad():
        
        if rank == 0 or rank =='cuda:0':
            pbar = tqdm(data_loader, file=sys.stdout)
        else:
            pbar = data_loader
            
        for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), target) in enumerate(pbar):
            input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)
            type_ids, spk_type_ids = type_ids.to(rank), type_ids.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            output, logit, seq_logits = model(input_ids=input_ids, attention_mask=att_mask,
                                              token_type_ids=type_ids, speaker_type_ids=spk_type_ids,
                                              is_inference=is_inference)

            loss = criterion(logit, target)
            acc = calc_acc(logit, target)

            sum_loss += loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            if rank == 0 or rank =='cuda:0':
                pbar.set_postfix(loss='{:.8f}, acc={:.4f}'.format(loss, acc))
            
            output_pred = logit.detach().cpu()
            true_label = target.detach().cpu()
            predicted = torch.concat([predicted, output_pred], dim=0)
            labels = torch.concat([labels, true_label], dim=0)
        if rank == 0 or rank =='cuda:0':
            pbar.close()

    total_loss = sum_loss / (batch_idx + 1)
    total_acc = sum_acc / (batch_idx * bs + mb_len)
    
    # predicted_probas = torch.sigmoid(predicted)[:, 1]
    predicted_probas = torch.softmax(predicted, dim=-1)[:, 1]
    predicted_labels = torch.where(predicted_probas >= label_frequency , 1, 0)
    labels_ = torch.where(labels == 1, 1, 0)
    
    predicted_probas = predicted_probas.numpy()
    predicted_labels = predicted_labels.numpy()
    labels_ = labels_.numpy()
    
    AUROC = roc_auc_score(labels_, predicted_probas)
    AUPRC = average_precision_score(labels_, predicted_probas)
    TH_ACC = accuracy_score(labels_, predicted_labels)
    RECALL = recall_score(labels_, predicted_labels)
    PRECISION = precision_score(labels_, predicted_labels)
    SPECIFICITY = specificity_score(labels_, predicted_labels)
    F1 = f1_score(labels_, predicted_labels)
    BRIER = brier_score_loss(labels_, predicted_probas)
    # CM = confusion_matrix(labels, predicted_labels)
    if is_inference:
        return total_loss, total_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, SPECIFICITY, F1, BRIER), (predicted_probas, labels)
    else:
        return total_loss, total_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, SPECIFICITY, F1, BRIER)

def inference(model, rank, criterion, data_loader, label_frequency, test_file_ids, pad_token_id=0):
    assert rank == 0 or rank == 'cuda:0'

    model.eval()
    sum_loss = sum_acc = 0
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
        for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), target) in enumerate(pbar):
            input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)
            type_ids, spk_type_ids = type_ids.to(rank), type_ids.to(rank)
            target = target.to(rank)
            mb_len = len(target)

            output, logit, seq_logits = model(input_ids=input_ids, attention_mask=att_mask,
                                              token_type_ids=type_ids, speaker_type_ids=spk_type_ids,
                                              is_inference=True)
            
            sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(rank)

            loss = criterion(logit, target)
            # loss = get_loss(rank, input_ids, output, target, criterion)
            acc = calc_acc(logit, target)

            sum_loss += loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            pbar.set_postfix(loss='{:.8f}, acc={:.4f}'.format(loss, acc))
            
            output_pred = logit.detach().cpu()
            true_label = target.detach().cpu()
            predicted = torch.concat([predicted, output_pred], dim=0)
            labels = torch.concat([labels, true_label], dim=0)

            input_ids = input_ids.detach().cpu()
            spk_type_ids = spk_type_ids.detach().cpu()
            for idx, (file_id, seq_len, seq_logit) in enumerate(zip(test_file_ids, sequence_lengths, seq_logits)):
                seq_length = seq_logit.size(0)
                
                file_ids = [file_id] * seq_length
                file_ids_list += file_ids

                label = [target[idx].item()] * seq_length
                label_per_token_logits += label
                # input_id = input_ids[idx][:seq_len]; input_ids_list = torch.concat([input_ids_list, input_id], dim=0)
                # spk_type_id = spk_type_ids[idx][:seq_len]; spk_type_ids_list = torch.concat([spk_type_ids_list, spk_type_id], dim=0)
                
                # logits = output[idx, :seq_len]; logits=logits.detach().cpu()
                # seq_logits_concated = torch.concat(seq_logit, dim=0).detach().cpu()
                predicted_token_logits = torch.concat([predicted_token_logits, seq_logit.detach().cpu()], dim=0)

            test_file_ids = test_file_ids[bs:]

        pbar.close()

    total_loss = sum_loss / (batch_idx + 1)
    total_acc = sum_acc / (batch_idx * bs + mb_len)
    
    # predicted_probas = torch.sigmoid(predicted)[:, 1]
    predicted_probas = torch.softmax(predicted, dim=-1)[:, 1]
    predicted_labels = torch.where(predicted_probas >= label_frequency , 1, 0)
    labels_ = torch.where(labels == 1, 1, 0)
    
    predicted_probas = predicted_probas.numpy()
    predicted_labels = predicted_labels.numpy()
    labels_ = labels_.numpy()

    file_ids_list = np.array(file_ids_list)
    # input_ids_list = input_ids_list.numpy().astype(np.int32)
    # spk_type_ids_list = spk_type_ids_list.numpy().astype(np.int32)

    predicted_token_logits = torch.softmax(predicted_token_logits, dim=-1)
    
    predicted_token_proba_0 = predicted_token_logits[:, 0]
    predicted_token_proba_0 = np.round(predicted_token_proba_0.numpy(), 8)

    predicted_token_proba_1 = predicted_token_logits[:, 1]
    predicted_token_proba_1 = np.round(predicted_token_proba_1.numpy(), 8)

    predicted_token_proba_2 = predicted_token_logits[:, 2]
    predicted_token_proba_2 = np.round(predicted_token_proba_2.numpy(), 8)

    # predicted_token_logits = torch.where(predicted_token_logits >= label_frequency , 1, 0)
    # predicted_token_logits = np.round(predicted_token_logits.numpy(), 8)
    label_per_token_logits = np.array(label_per_token_logits).astype(np.int32)

    return (predicted_probas, labels_), (file_ids_list, predicted_token_proba_0, predicted_token_proba_1, predicted_token_proba_2, label_per_token_logits)

# def train(device, model, criterion, optimizer, scheduler, epochs, save_path,
#           train_loader, valid_loader=None, save_term=256, label_frequency=0.5):
#     """
#     :param model: your model
#     :param device: your device(cuda or cpu)
#     :param optimizer: your optimizer
#     :param criterion: loss function
#     :param epochs: train epochs
#     :param save_path : checkpoint path
#     :param train_loader: train dataset
#     :param valid_loader: valid dataset
#     """
#     model.to(device)
#     for epoch in range(1, epochs + 1):
#         model.train()
        
#         loss_lst = []
#         acc_lst = []
#         bs = train_loader.batch_size

#         # in notebook
#         # pabr = notebook.tqdm(enumerate(train_loader), file=sys.stdout)

#         # in interpreter
#         pbar = tqdm(train_loader, file=sys.stdout)
#         for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), target) in enumerate(pbar):
#             input_ids, att_mask = input_ids.to(device), att_mask.to(device)
#             type_ids, spk_type_ids = type_ids.to(device), type_ids.to(device)
#             target = target.to(device)
#             mb_len = len(target)

#             optimizer.zero_grad()
#             output = model(input_ids=input_ids, attention_mask=att_mask,
#                            token_type_ids=type_ids, speaker_type_ids=spk_type_ids)
#             # output = get_each_output(output)
#             loss = criterion(output, target)
#             acc = calc_acc(output, target)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             loss_lst.append(loss.item()); acc_lst.append(acc)
#             pbar.set_postfix(epoch=f'{epoch}/{epochs}', loss='{:.4f}, acc={:.4f}'.format(np.mean(loss_lst), np.sum(acc_lst) / (batch_idx * bs + mb_len)))
            
#             if batch_idx != 0 and batch_idx % save_term == 0:
#                 torch.save({
#                     'model_state_dict': model.module.state_dict(),
#                     'epoch': epoch,
#                     'batch_idx': batch_idx
#                 }, os.path.join(save_path, f'checkpoint_{epoch}_{batch_idx}.tar'))
#         pbar.close()

#         if valid_loader is not None:
#             valid_loss, valid_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER) = evaluate(model, 
#                                                                                                    device, 
#                                                                                                    criterion, 
#                                                                                                    valid_loader,
#                                                                                                    label_frequency)
#             print("valid loss : {:.6f}".format(valid_loss))
#             print("valid acc : {:.3f}".format(valid_acc))
#             print("valid acc(th) : {:4f}".format(TH_ACC))
#             print("valid AUROC : {:.4f}".format(AUROC))
#             print("valid AUPRC : {:.4f}".format(AUPRC))
#             print("valid Recall : {:4f}".format(RECALL))    
#             print("valid Precision : {:.4f}".format(PRECISION))
#             print("valid F1_score : {:.4f}".format(F1))
#             print("valid Brier : {:4f}".format(BRIER))
#         print()

#         if epoch % 1 == 0:
#             torch.save({
#                 'model_state_dict': model.module.state_dict(),
#                 'epoch': epoch,
#                 'batch_idx': batch_idx
#             }, os.path.join(save_path, f'checkpoint_{epoch}_{batch_idx}.tar'))
#     return model

# def evaluate(model, device, criterion, data_loader, label_frequency, is_inference=False):
#     """
#     :param model: your model
#     :param device: your device(cuda or cpu)
#     :param criterion: loss function
#     :param data_loader: valid or test Datasets
#     :return: (valid or test) loss and acc
#     """
#     model.eval()
#     sum_loss = sum_acc = 0
#     bs = data_loader.batch_size
    
#     predicted = torch.tensor([])
#     labels = torch.tensor([])

#     with torch.no_grad():
#         # in notebook
#         # pabr = notebook.tqdm(enumerate(valid_loader), file=sys.stdout)

#         # in interpreter
#         pbar = tqdm(data_loader, file=sys.stdout)
#         for batch_idx, ((input_ids, att_mask, type_ids, spk_type_ids), target) in enumerate(pbar):
#             input_ids, att_mask = input_ids.to(device), att_mask.to(device)
#             type_ids, spk_type_ids = type_ids.to(device), type_ids.to(device)
#             target = target.to(device)
#             mb_len = len(target)

#             output = model(input_ids=input_ids, attention_mask=att_mask,
#                            token_type_ids=type_ids, speaker_type_ids=spk_type_ids)
#             # output = get_each_output(output)
#             loss = criterion(output, target)
#             acc = calc_acc(output, target)

#             sum_loss += loss.item()
#             sum_acc += acc

#             loss = sum_loss / (batch_idx + 1)
#             acc = sum_acc / (batch_idx * bs + mb_len)
#             pbar.set_postfix(loss='{:.4f}, acc={:.4f}'.format(loss, acc))
            
#             if type(output) is list:
#                 logits = [o.detach().cpu() for o in output]
#                 logits = torch.cat(logits, dim=0)
#             else:
#                 logits = output.detach().cpu()
#             true_label = target.detach().cpu()
#             predicted = torch.concat([predicted, logits], dim=0)
#             labels = torch.concat([labels, true_label], dim=0)
#         pbar.close()

#     total_loss = sum_loss / (batch_idx + 1)
#     total_acc = sum_acc / (batch_idx * bs + mb_len)
    
#     # predicted_probas = torch.sigmoid(predicted)[:, 1]
#     predicted_probas = torch.softmax(predicted, dim=-1)[:, 1]
#     predicted_labels = torch.where(predicted_probas >= label_frequency , 1, 0)
    
#     predicted_probas = predicted_probas.numpy()
#     predicted_labels = predicted_labels.numpy()
#     labels = labels.numpy()
    
#     AUROC = roc_auc_score(labels, predicted_probas)
#     AUPRC = average_precision_score(labels, predicted_probas)
#     TH_ACC = accuracy_score(labels, predicted_labels)
#     RECALL = recall_score(labels, predicted_labels)
#     PRECISION = precision_score(labels, predicted_labels)
#     F1 = f1_score(labels, predicted_labels)
#     BRIER = brier_score_loss(labels, predicted_probas)
#     # CM = confusion_matrix(labels, predicted_labels)
#     if is_inference:
#         return total_loss, total_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER), (predicted_probas, labels)
#     else:
#         return total_loss, total_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER)

def main():
    pass


if __name__ == "__main__":
    main()
