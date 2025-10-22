import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
    
class MyDataset(Dataset):
    def __init__(self, 
                 data, 
                 tokenizer, 
                 max_length=512, 
                 padding='max_length',
                 speaker_num=4,
                 class_num=2,):
        super(MyDataset, self).__init__()

        self.data = data
        self.label = data.label.values
        self.text = data.text.values
        # self.label_tag = {'하':0, '중':0, '상':1, '최상':1}
        
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = 'pt'
        self.return_token_type_ids = True
        self.return_attention_mask = True
        
        self.speaker_num = speaker_num
        self.class_num = class_num
        
        self.speaker_token_num = {tokenizer.convert_tokens_to_ids(f'[SPK{i}]'):i for i in range(speaker_num)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ## sentence ##
        text = self.text[idx] # add CLS token for sequence classification
        tokenizer_output = self.tokenizer(text, max_length=self.max_length, padding=self.padding,
                                          return_tensors=self.return_tensors, truncation=True,
                                          return_token_type_ids=self.return_token_type_ids,
                                          return_attention_mask=self.return_attention_mask)

        input_ids = tokenizer_output['input_ids'][0]
        att_mask = tokenizer_output['attention_mask'][0]
        type_ids = tokenizer_output['token_type_ids'][0]
        spk_type_ids = []
        spk_type_id  = 0
        for id in input_ids:
            id = id.item()
            if id in list(range(self.speaker_num)): # [PAD], [UNK], [CLS], [SEP]는 spk_type == 0
                spk_type_ids.append(0)
                continue
            if id in self.speaker_token_num.keys():
                spk_type_id = self.speaker_token_num[id]
            spk_type_ids.append(spk_type_id)
        spk_type_ids = torch.tensor(spk_type_ids).long()

        ## label ##
        ## 0: blank | 1: False | 0: True ##
        y = self.label[idx]
        if y: # y is True
            target = torch.tensor([0, 2]).long()
        else: # y is False
            target = torch.tensor([0, 1]).long()
        target_length = torch.tensor(len(target)).long()
        return (input_ids, att_mask, type_ids, spk_type_ids), (target, target_length)

def main():
    pass

if __name__ == "__main__":
    main()