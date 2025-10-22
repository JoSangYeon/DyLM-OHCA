import os
import sys
import glob
import pickle
import librosa
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (Dataset, DataLoader, 
                              random_split, RandomSampler)

from transformers import AutoProcessor, ASTModel

DEVICE = 'cuda'

class MyDataset(Dataset):
    def __init__(self, 
                 root_path,
                 wav_list, 
                 processor,
                 sr=16000):
        super(MyDataset, self).__init__()
        
        self.root_path = root_path
        self.wav_list = wav_list
        
        self.processor = processor

        self.sr = sr

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        # wav_path = os.path.join(self.root_path, self.wav_list[idx])
        wav_path = self.wav_list[idx]

        wav, sr = librosa.load(wav_path)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)

        wav = self.processor(wav, sampling_rate=self.sr, return_tensors="pt")
        wav = wav.input_values[0] # Size(1024, 128)

        return wav

def make_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

def load_model(link="MIT/ast-finetuned-audioset-10-10-0.4593"):
    processor = AutoProcessor.from_pretrained(link)
    model = ASTModel.from_pretrained(link)
    return model, processor

def process(device, model, dataloder, save_path, wav_list):
    model.to(device); model.eval()
    bs = dataloder.batch_size

    with torch.no_grad():
        pbar = tqdm(dataloder, file=sys.stdout)
        for batch_idx, wav in enumerate(pbar):
            wav = wav.to(device)

            output = model(wav) # (batch, 1214, 768) | (batch, 768)
            cls_embed = output[1]

            save_npy(cls_embed, save_path, wav_list)
            wav_list = wav_list[bs:]


def save_npy(cls_embed, save_path, wav_list):
    for bs, (path, embed) in enumerate(zip(wav_list, cls_embed)):
        file_name = os.path.basename(path).split('.')[0]+'.npy'
        npy_save_path = os.path.join(save_path, file_name)

        embed_npy = embed.detach().cpu().numpy()
        np.save(npy_save_path, embed_npy)

def main():
    model_link = "MIT/ast-finetuned-audioset-10-10-0.4593"

    root_path = 'toy_sample'
    save_path = make_dir(os.path.join(root_path, 'wav_npy'))
    wav_list = glob.glob(os.path.join(root_path, '*.wav'))

    model, processor = load_model(link=model_link)
    dataset = MyDataset(root_path, wav_list, processor)
    dataloder = DataLoader(dataset, batch_size=4)

    process(DEVICE, model, dataloder, save_path, wav_list)


if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=3 python transform_tensor2npy.py