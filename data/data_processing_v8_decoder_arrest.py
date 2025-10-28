import os
import glob
import shutil
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import pandas as pd
import psutil
import pdb
import numpy as np
import ast

cpu_set   = 0
num_cpu   = 96
p = psutil.Process()
p.cpu_affinity( [int(num_cpu*cpu_set + i) for i in range(30, num_cpu)] )

''' 
label 1
증상없음 -> 0,
심정지 or 호흡정지 -> 1,
‘심정지 or 호흡정지’가 없고 나머지 증상중 하나라도 있으면 -> 2

label 2
['흉통', '가슴불편감', '실신', '호흡곤란', '심계향진'] 중 하나
'''
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=30 python data_processing_v8_decoder.py

json_file_path='/storage/projects/NIA/datasets/json/'
wav_file_path='/storage/projects/NIA/datasets/wav/'
all_json_files = sorted(glob.glob(json_file_path+'*.json'))
all_wav_files = sorted(glob.glob(wav_file_path+'original/*.wav'))

audio_type='wav'

df_dict = {'json_path':[],'wav_path':[], 'id':[], 'symptom':[],'label':[],'label1':[],'label2':[]}


emergency_symptoms_list = ['흉통', '절단', '압궤손상', '의식장애', '호흡곤란', '호흡정지','경련/발작', '실신', '토혈', '혈변', '저체온증', '마비', '기도이물']
emergency_symptoms_list_sub1 =['얼굴마비', '구음장애', '팔 위약/ 마비', '다리 위약/마비', '의식장애', '어지러움', '두통', '경련/발작', '실신']
emergency_symptoms_list_sub2 = ['흉통', '가슴불편감', '실신', '호흡곤란', '심계향진']
emergency_symptoms_list.append(emergency_symptoms_list_sub1)
emergency_symptoms_list.append(emergency_symptoms_list_sub2)

emergency_symptoms_list2 = ['흉통', '가슴불편감', '실신', '호흡곤란', '심계향진']

def check_elements(a, b):
    for sub_list in b:
        count=0 
        for elem in a:
            if elem in b:
                return True
            if isinstance(sub_list, list):
                if elem in sub_list:
                    count+=1
                    if count > 1:
                        return True
    return False

for file_path in all_json_files:
    filename=file_path.split('/')[-1]
    wav_name=filename.split('.json')[0]+'.wav'
    if filename.endswith('.json') and os.path.exists(os.path.join(wav_file_path,'original',filename.split('.json')[0]+'.wav')):
        with open(file_path, 'r') as f:
            
            data = json.load(f)

            if isinstance(data['symptom'],list) and 'symptom' in data.keys():
                # Comments #
                utterances = data['utterances']
                text = []
                
                # Create label1
                if isinstance(data['symptom'], list):
                    if '심정지' in data['symptom']:
                        label1 = 1
                    elif check_elements(data['symptom'],emergency_symptoms_list):
                        label1 = 2
                    else:
                        label1 = 0
                else:
                    label1 = 0

                label2 = check_elements(data['symptom'],emergency_symptoms_list2)
                label = check_elements(data['symptom'],['심정지'])

                # Temporary Label 
                df_dict['json_path'].append(file_path)
                df_dict['wav_path'].append(os.path.join(wav_file_path,'original',filename.split('.json')[0]+'.wav'))
                df_dict['id'].append(data['_id'])
                df_dict['symptom'].append(data['symptom'])


                df_dict['label'].append(label)
                df_dict['label1'].append(label1)
                df_dict['label2'].append(label2)


data_df = pd.DataFrame(df_dict)


train_list, test_list=train_test_split(data_df, test_size=0.2, stratify=data_df['label1'],random_state=42)
train_list, valid_list=train_test_split(train_list, test_size=0.2, stratify=train_list['label1'],random_state=42)

# 분할 정보를 나타내는 열 추가
data_df['split'] = 'None'  # 초기값으로 'None' 할당

train_indices = train_list.index
valid_indices = valid_list.index
test_indices = test_list.index


data_df.loc[train_indices, 'split'] = 'train'
data_df.loc[valid_indices, 'split'] = 'valid'
data_df.loc[test_indices, 'split'] = 'test'


data_df.to_csv('data_df.csv', index=False)

train_df=data_df.loc[train_indices]
valid_df=data_df.loc[valid_indices]
test_df=data_df.loc[test_indices]

train_df.to_csv('train_df.csv', index=False)
valid_df.to_csv('valid_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

def make_label(root_path, df, mode, start_sentence=0,window_size=0, max_sentence=85):

    path = os.path.join(root_path,'json')
    audio_path = os.path.join(root_path, 'wav', 'original')
    # audio_slide_path = os.path.join(root_path, 'wav', 'original_slide')

    df_dict = {'id':[], 'json_file_path':[], 'wav_original_file_path':[] , 'endAt':[], 'label':[], 'label1':[], 'label2':[], 'text':[]}
    for num_s in range(max_sentence):
        df_dict[f'S{num_s}_Speaker']=[]
        df_dict[f'S{num_s}']=[]
        df_dict[f'S{num_s}_startAt']=[]
        df_dict[f'S{num_s}_endAt']=[]

    emergency_symptoms_list = ['흉통', '절단', '압궤손상', '의식장애', '호흡곤란', '호흡정지','경련/발작', '실신', '토혈', '혈변', '저체온증', '마비', '기도이물']
    emergency_symptoms_list_sub1 =['얼굴마비', '구음장애', '팔 위약/ 마비', '다리 위약/마비', '의식장애', '어지러움', '두통', '경련/발작', '실신']
    emergency_symptoms_list_sub2 = ['흉통', '가슴불편감', '실신', '호흡곤란', '심계향진']
    emergency_symptoms_list.append(emergency_symptoms_list_sub1)
    emergency_symptoms_list.append(emergency_symptoms_list_sub2)

    emergency_symptoms_list2 = ['흉통', '가슴불편감', '실신', '호흡곤란', '심계향진']

    def check_elements(a, b):
        for sub_list in b:
            count=0 
            for elem in a:
                if elem in b:
                    return True
                if isinstance(sub_list, list):
                    if elem in sub_list:
                        count+=1
                        if count > 1:
                            return True
        return False

    json_file_path=df['json_path']
    original_file_path=df['wav_path']

    for n, (filename,wav_path) in tqdm(enumerate(zip(json_file_path,original_file_path))):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)

            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data['symptom'],list) and 'symptom' in data.keys():
                    # Comments #
                    utterances = data['utterances']
                    text = []
                    startAt=[]
                    speaker = []
                    spk_text =[]
                    start_time = []
                    end_time = []
                    t=0
                    # l=0
                if isinstance(data['symptom'],list) and 'symptom' in data.keys():
                    # Comments #
                    utterances = data['utterances']
                    text = []
                    # Create label1
                    if isinstance(data['symptom'], list):
                        if '심정지' in data['symptom']:
                            label1 = 1
                        elif check_elements(data['symptom'],emergency_symptoms_list):
                            label1 = 2
                        else:
                            label1 = 0
                    else:
                        label1 = 0

                    label2 = check_elements(data['symptom'],emergency_symptoms_list2)
                    label = check_elements(data['symptom'],['심정지'])

                    for i, each in enumerate(utterances):
                        # if mode == 'test' and each['endAt'] > 120000: 
                        # if mode == 'test' and each['endAt'] > 90000: 
                        if mode == 'test' and each['endAt'] > 60000: 
                            i=i-1
                            break
                        t=t+1
                        start_time.append(each['startAt'])
                        end_time.append(each['endAt'])

                        text.append(each['text'])

                        speaker.append(each['speaker'])
                        spk_text+=[f"[SPK{each['speaker']}] {each['text']}"]

                        if i < start_sentence:
                            text=[' '.join(text)]
                            continue

                        if len(text) < window_size:
                            continue

                    if not end_time:
                        continue

                    df_dict['json_file_path'].append(file_path)
                    # if mode == 'test':
                    #     df_dict['wav_original_file_path'].append(root_path+'wav/original_cut/{}'.format(os.path.splitext(os.path.basename(wav_path))[0]+'_cut.wav'))
                    # else:
                    #     df_dict['wav_original_file_path'].append(wav_path)
                    df_dict['wav_original_file_path'].append(wav_path)

                        # l=l+1
                    # pdb.set_trace()
                    for num in range(i+1):
                        df_dict[f'S{num}_Speaker'].append(int(speaker[num]))
                        df_dict[f'S{num}'].append(text[num])
                        df_dict[f'S{num}_startAt'].append(int(start_time[num]))
                        df_dict[f'S{num}_endAt'].append(int(end_time[num]))


                    for na_num in range(i+1,max_sentence):
                        df_dict[f'S{na_num}_Speaker'].append(None)
                        df_dict[f'S{na_num}'].append(None)
                        df_dict[f'S{na_num}_startAt'].append(None)
                        df_dict[f'S{na_num}_endAt'].append(None)
                    
                    df_dict['text']+=[' '.join(spk_text)]
                    df_dict['endAt'].append(int(end_time[-1]))
                    df_dict['id'].append(data['_id'])
                    df_dict['label'].append(label)
                    df_dict['label1'].append(label1)
                    df_dict['label2'].append(label2)
                    # df_dict['length'].append(t)


    # pdb.set_trace()
    data_df = pd.DataFrame(df_dict)
    return data_df
# id,json_file_path,wav_original_file_path,wav_slide_file_path,text,startAt,endAt,label


# ##################### 120 seconds #####################
# print('###########make_toy_data_label###########')
# train=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=train_df.iloc[:1000], mode='train',start_sentence=0,window_size=0, max_sentence=117)
# train.to_csv('toy_data_json_audio_data_decoder_time_arrest.csv', index=False)


# # print('###########make_train_label###########')
# train=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=train_df, mode='train',start_sentence=0,window_size=0, max_sentence=117)
# train.to_csv('train_json_audio_data_decoder_time_arrest.csv', index=False)


# print('###########make_valid_label###########')
# valid=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=valid_df, mode='test',start_sentence=0,window_size=0, max_sentence=117)
# valid.to_csv('valid_json_audio_data_decoder_time_arrest.csv', index=False)


# print('###########make_test_label###########')

# test=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=test_df,mode='test',start_sentence=0,window_size=0, max_sentence=117)
# test.to_csv('test_json_audio_data_decoder_time_arrest.csv', index=False)


# ##################### 90 seconds #####################
# print('###########make_toy_data_label###########')
# train=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=train_df.iloc[:1000], mode='train',start_sentence=0,window_size=0, max_sentence=117)
# train.to_csv('toy_data_json_audio_data_decoder_time_arrest_90s.csv', index=False)


# print('###########make_valid_label###########')
# valid=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=valid_df, mode='test',start_sentence=0,window_size=0, max_sentence=117)
# valid.to_csv('valid_json_audio_data_decoder_time_arrest_90s.csv', index=False)


# print('###########make_test_label###########')

# test=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=test_df,mode='test',start_sentence=0,window_size=0, max_sentence=117)
# test.to_csv('test_json_audio_data_decoder_time_arrest_90s.csv', index=False)

##################### 60 seconds #####################
print('###########make_toy_data_label###########')
train=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=train_df.iloc[:1000], mode='train',start_sentence=0,window_size=0, max_sentence=117)
train.to_csv('toy_data_json_audio_data_decoder_time_arrest_60s.csv', index=False)


print('###########make_valid_label###########')
valid=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=valid_df, mode='test',start_sentence=0,window_size=0, max_sentence=117)
valid.to_csv('valid_json_audio_data_decoder_time_arrest_60s.csv', index=False)


print('###########make_test_label###########')

test=make_label(root_path='/storage/projects/NIA/datasets/cau/',df=test_df,mode='test',start_sentence=0,window_size=0, max_sentence=117)
test.to_csv('test_json_audio_data_decoder_time_arrest_60s.csv', index=False)



print('\ntrain/valid/test 데이터의 수')
print((len(train),len(valid),len(test)))
print('train/valid/test 데이터의 label 응급비율')
print((sum(train['label']==True)/len(train),
sum(valid['label']==True)/len(valid),
sum(test['label']==True)/len(test)))

print('train/valid/test 데이터의 label1 응급비율')
print('0:')
print((sum(train['label1']==0)/len(train),
sum(valid['label1']==0)/len(valid),
sum(test['label1']==0)/len(test)))

print('1:')
print((sum(train['label1']==1)/len(train),
sum(valid['label1']==1)/len(valid),
sum(test['label1']==1)/len(test)))

print('2:')
print((sum(train['label1']==2)/len(train),
sum(valid['label1']==2)/len(valid),
sum(test['label1']==2)/len(test)))


print('train/valid/test 데이터의 label2 응급비율')
print((sum(train['label2']==True)/len(train),
sum(valid['label2']==True)/len(valid),
sum(test['label2']==True)/len(test)))