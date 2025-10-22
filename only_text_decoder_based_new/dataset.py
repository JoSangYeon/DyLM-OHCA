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
        self.label = data.label
        self.text = data.text
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
        text = self.text[idx]
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
        # y = self.label_tag[self.data.label[idx]]
        # y = torch.tensor(y).long()
        y = self.label[idx]
        y = torch.tensor(y).long()
        # y = F.one_hot(y, num_classes=self.class_num).float()
        # print(input_ids.shape, att_mask.shape, type_ids.shape, spk_type_ids.shape)
        
        return (input_ids, att_mask, type_ids, spk_type_ids), y

    def shape(self):
        return self.data.shape