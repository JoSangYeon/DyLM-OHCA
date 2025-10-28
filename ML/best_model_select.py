import numpy as np
import pandas as pd
import os, sys
import pickle
import yaml
from imblearn.metrics import specificity_score

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, brier_score_loss, f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

# libraries for benchmark models

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import pdb

from tqdm import tqdm
import joblib
import psutil
from datetime import datetime
now = datetime.now()

import argparse

from utils import calc_metric, calc_metric_roc_th
import multiprocessing as mp

import random
import sklearn

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--window_size', type=int, default=0, help="window_size")
    parser.add_argument('--prj_dir', default='/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/benchmarks') #'/storage/personal/chkim/NLP/Topic_model/bow
    parser.add_argument('--cpu_start', '-c', type=int, default=0, help="bootstrapping number")
    available_benchmarks =['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest']
    parser.add_argument('--benchmarks', nargs='+', choices=available_benchmarks, default=[], help="List of benchmarks to run")
    parser.add_argument('--time_cut', type=int, default=120, help="time cut for tf-idf")
    return parser.parse_args()

args = parse_args()
#cpu affinity 
p = psutil.Process()
p.cpu_affinity(range(int(args.cpu_start),int(args.cpu_start)+25))
print(p.cpu_affinity())


def moving_average_segments(df, predictions, window_size):
    moving_avg_segments = []
    ids = df['id'].unique()  # DataFrame에서 유일한 id 값을 추출
    
    for id in tqdm(ids, desc="segment_processing"):
        # 현재 id에 해당하는 s_id 세그먼트 필터링
        id_mask = df['id'] == id
        id_predictions = predictions[id_mask, :]  # 해당 id의 예측 확률 값들
        
        if window_size == 0:
            # 이동 평균을 계산하지 않고 원래의 예측 값만 리스트에 저장
            moving_avg_segments.append(id_predictions)
        else:
            # 해당 id의 세그먼트들에 대한 이동 평균 계산
            moving_avg_id = []
            num_segments = id_predictions.shape[0]  # 현재 id에 속한 세그먼트 개수
            
            for i in range(num_segments):
                start = max(0, i - window_size + 1)  # 윈도우의 시작점 (음수가 되지 않도록)
                end = i + 1  # 윈도우의 끝점
                window_avg = id_predictions[start:end, :].mean(axis=0)  # 해당 윈도우 내의 평균 계산
                moving_avg_id.append(window_avg)
            
            moving_avg_segments.append(np.array(moving_avg_id))  # 계산된 이동 평균 값을 리스트에 저장
    
    return moving_avg_segments  # 모든 id에 대한 이동 평균 값 반환

def calculate_column_statistic(arrays, column_index, method="max"):
    if method == "max":
        return np.array([arr[:, column_index].max() for arr in arrays])
    elif method == "average":
        return np.array([arr[:, column_index].mean() for arr in arrays])
    else:
        raise ValueError("Method must be 'max' or 'average'")

# -- Load Data --
if args.time_cut ==120:
    tr_data = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','train_json_audio_data_decoder_time_arrest.csv'), index_col=0)
    va_data = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','valid_json_audio_data_decoder_time_arrest.csv'), index_col=0)
elif args.time_cut == 90:
    tr_data = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','train_json_audio_data_decoder_time_arrest.csv'), index_col=0)
    va_data = pd.read_csv(os.path.join('/home/myhwang/119_AMIGO/','valid_json_audio_data_decoder_time_arrest_90s.csv'), index_col=0)
elif args.time_cut == 60:
    tr_data = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','train_json_audio_data_decoder_time_arrest.csv'), index_col=0)
    va_data = pd.read_csv(os.path.join('/home/myhwang/119_AMIGO/','valid_json_audio_data_decoder_time_arrest_60s.csv'), index_col=0)

# te_data = pd.read_csv(os.path.join(args.data_dir,'test_json_audio_data_decoder_time_arrest.csv'), index_col=0)

# -- Extract index --
train_index = tr_data.index
valid_index = va_data.index
# test_index = te_data.index

# bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf.csv'), index_col='s_id') # 's_id'를 인덱스로 설정
# tr_bow_df = bow_df.loc[train_index]
# va_bow_df = bow_df[bow_df['id'].isin(valid_index)]
# te_bow_df = bow_df[bow_df['id'].isin(test_index)]
# -- Load Data --
if args.time_cut ==120:
    tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train.csv'), index_col='id')
    va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf.csv'), index_col='s_id')
elif args.time_cut == 90:
    tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train.csv'), index_col='id')
    va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf(tr)_90s.csv'), index_col='s_id')
elif args.time_cut == 60:
    tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train.csv'), index_col='id')
    va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf(tr)_60s.csv'), index_col='s_id')
# # mecab
# tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train_mecab.csv'), index_col='id')
# va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf(tr)_mecab.csv'), index_col='s_id')

# tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train_new.csv'), index_col='id')
# va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf(tr)_mecab_new.csv'), index_col='s_id')



# te_bow_df = pd.read_csv(os.path.join(args.bow_dir,'test_tf_idf.csv'), index_col='s_id')

# data = pd.read_csv(args.data_dir, index_col=0) # [07.04]label 1: 응급실 방문 권고 ,  나머지 0
# bow_df = pd.read_csv(args.bow_dir, index_col=0)

#label option 
X_tr = tr_bow_df.loc[:, ~tr_bow_df.columns.isin(['Unnamed: 0', 'id'])]
X_va = va_bow_df.loc[:, ~va_bow_df.columns.isin(['Unnamed: 0', 'id', 's_id'])]
# X_te = te_bow_df.loc[:, ~te_bow_df.columns.isin(['Unnamed: 0', 'id', 's_id'])]
# X = bow_df.iloc[:,:-1]  # last column = sum

X_va_id = va_bow_df.loc[:, ['id']]
# id랑 샘플수 확인
# X_te_id = te_bow_df.loc[:, ['id']]
# X_te_id = te_bow_df.loc[:,'id']



Y_tr = tr_data.loc[:, 'label1']
Y_va = va_data.loc[:, 'label1']
# Y_te = te_data.loc[:, args.label]

BN_Y_va = Y_va.apply(lambda x: 1 if x == 1 else 0)
# BN_Y_te = Y_te.apply(lambda x: 1 if x == 1 else 0)
# # tabular_list=['SEX', 'AGE_MONTH', 'MNTPULSECNT', 'MNTBRETHCNT', 'CHOSBDTP', 
# #    'CHOSPATH_0_직접_내원','CHOSPATH_1_외부에서_전원', 'CHOSPATH_2_외래에서_의뢰', 
# #    'CHOSWAY_0_기타자동차','CHOSWAY_1_119구급차', 'CHOSWAY_2_의료기관구급차', 'CHOSWAY_3_기타구급차',
# #    'CHOSWAY_4_항공이송', 'CHOSWAY_5_도보', 'CHOSWAY_6_경찰차 등 공공차량','CHOSWAY_7_기타',
# # #    'NRS','Wong','FLACC','VAS','CRIES'
# #    ]
# if args.pain_score:
#     tabular_list=['SEX', 'AGE_MONTH', 'MNTPULSECNT', 'MNTBRETHCNT', 'CHOSBDTP', 
#     'CHOSPATH_0_직접_내원','CHOSPATH_1_외부에서_전원', 'CHOSPATH_2_외래에서_의뢰', 
#     'CHOSWAY_0_기타자동차','CHOSWAY_1_119구급차', 'CHOSWAY_2_의료기관구급차', 'CHOSWAY_3_기타구급차',
#     'CHOSWAY_4_항공이송', 'CHOSWAY_5_도보', 'CHOSWAY_6_경찰차 등 공공차량','CHOSWAY_7_기타',
#     #    'NRS','Wong','FLACC','VAS','CRIES'
#     # 'IS_PAIN',
#     'PAIN_SCORE',
#     # 'PAIN_SCALE',
#     ]
# else:
#     tabular_list=['SEX', 'AGE_MONTH', 'MNTPULSECNT', 'MNTBRETHCNT', 'CHOSBDTP', 
#    'CHOSPATH_0_직접_내원','CHOSPATH_1_외부에서_전원', 'CHOSPATH_2_외래에서_의뢰', 
#    'CHOSWAY_0_기타자동차','CHOSWAY_1_119구급차', 'CHOSWAY_2_의료기관구급차', 'CHOSWAY_3_기타구급차',
#    'CHOSWAY_4_항공이송', 'CHOSWAY_5_도보', 'CHOSWAY_6_경찰차 등 공공차량','CHOSWAY_7_기타',
#     #    'NRS','Wong','FLACC','VAS','CRIES'
#     # 'IS_PAIN','PAIN_SCORE',
#     # 'PAIN_SCALE',
#     ]
# X_tabular= data.loc[:, tabular_list]

print('X shape: ', X_tr.shape)
# print('X_tabular shape: ', X_tabular.shape)
print('Y shape: ', Y_tr.shape)




benchmark_list = args.benchmarks
# benchmark_list = ['Random Forest'] 
# benchmark_list = ['Gradient Boosting'] 
# benchmark_list = ['Logistic Regression']

# benchmark_list = ['XGBoost', 'Random Forest'] 
# benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest'] 


OUT_ITERATION=10
FINAL_RESULTS = np.zeros([OUT_ITERATION, len(benchmark_list), 8])

print("train, val data shape", X_tr.shape, Y_tr.shape, X_va.shape, Y_va.shape)


results = np.zeros([len(benchmark_list), 8])

benchmarks = {}

    
# print("test data shape", X_te.shape, Y_te.shape)


filename = os.path.join(args.prj_dir +'/scaler.sav')
with open (filename, 'rb') as f:
    scaler = pickle.load(f)
print("scaler loaded")
scaled_X_tr = pd.DataFrame(scaler.transform(X_tr), index=X_tr.index) 
scaled_X_va = pd.DataFrame(scaler.transform(X_va), index=X_va.index)
# scaled_X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index)
# X_tr, X_va, X_te = scaled_X_tr.copy(), scaled_X_va.copy(), scaled_X_te.copy()
X_tr, X_va = scaled_X_tr.copy(), scaled_X_va.copy()
    
# -- copy original data --

label_frequency = float(Y_tr[Y_tr==1].sum() / len(Y_tr))


print("label_frequency:", label_frequency)


import os
import pickle
from sklearn.metrics import roc_auc_score




# 경로 설정
PRJ_DIR = args.prj_dir
window_size = args.window_size


# 실행 예시 (모델별로 호출)
best_models1 = {}
best_models2 = {}


from joblib import Parallel, delayed
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
import pickle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def evaluate_single_file(modelname, file, X_va, X_va_id, BN_Y_va, window_size, PRJ_DIR):
    with open(os.path.join(PRJ_DIR, file), 'rb') as f:
        loaded_model = pickle.load(f)

    # 모델 성능 평가
    tmp_pred = loaded_model.predict_proba(X_va)  # predict_prob 2d array
    tmp_pred = moving_average_segments(X_va_id, tmp_pred, window_size)

    max_pred = calculate_column_statistic(tmp_pred, 1, 'max')
    average_pred = calculate_column_statistic(tmp_pred, 1, 'average')
    
    tmp_auroc1 = roc_auc_score(np.asarray(BN_Y_va), max_pred)
    tmp_auroc2 = roc_auc_score(np.asarray(BN_Y_va), average_pred)

    return tmp_auroc1, tmp_auroc2, loaded_model, file

def evaluate_model_files(modelname, files, X_va, X_va_id, BN_Y_va, window_size, PRJ_DIR):
    # 병렬 처리로 각 모델 파일을 평가
    with Parallel(n_jobs=-1, backend="loky", prefer="processes") as parallel:
        results = parallel(
            delayed(evaluate_single_file)(modelname, file, X_va, X_va_id, BN_Y_va, window_size, PRJ_DIR)
            for file in tqdm(files, desc=f"Processing {modelname}")
            if file.startswith(modelname)
        )


    # 각 모델 파일들 중 최고 AUROC를 가진 모델 선택
    best_auroc1 = 0.0
    best_model1 = None
    best_file1 = None
    
    best_auroc2 = 0.0
    best_model2 = None
    best_file2 = None
    
    for tmp_auroc1, tmp_auroc2, loaded_model, file in results:
        if tmp_auroc1 > best_auroc1:
            best_auroc1 = tmp_auroc1
            best_model1 = loaded_model
            best_file1 = file

        if tmp_auroc2 > best_auroc2:
            best_auroc2 = tmp_auroc2
            best_model2 = loaded_model
            best_file2 = file

    # 최종 선택된 베스트 모델 저장 (Max)
    if best_model1:
        # filename = '{}/best_{}_w{}_{}.sav'.format(PRJ_DIR, modelname, window_size, 'max')
        filename = '{}/best_{}_w{}_{}(train_idf)_{}.sav'.format(PRJ_DIR, modelname, window_size, 'max', args.time_cut)
        with open(filename, 'wb') as f:
            pickle.dump(best_model1, f)
        print(f'Best max model for {modelname} from file {best_file1} saved with AUROC: {best_auroc1}')

    # 최종 선택된 베스트 모델 저장 (Average)
    if best_model2:
        # filename = '{}/best_{}_w{}_{}.sav'.format(PRJ_DIR, modelname, window_size, 'average')
        filename = '{}/best_{}_w{}_{}(train_idf)_{}.sav'.format(PRJ_DIR, modelname, window_size, 'average', args.time_cut)
        with open(filename, 'wb') as f:
            pickle.dump(best_model2, f)
        print(f'Best average model for {modelname} from file {best_file2} saved with AUROC: {best_auroc2}')
    
    return (best_model1, best_auroc1), (best_model2, best_auroc2)


for modelname in benchmark_list:
    (best_model1, best_auroc1), (best_model2, best_auroc2) = evaluate_model_files(
        modelname=modelname, 
        files=os.listdir(PRJ_DIR), 
        X_va=X_va, 
        X_va_id=X_va_id, 
        BN_Y_va=BN_Y_va, 
        window_size=window_size, 
        PRJ_DIR=PRJ_DIR
    )
    
    # if best_model1:
    #     best_models1[modelname] = best_model1
    # if best_model2:
    #     best_models2[modelname] = best_model2

# # 베스트 모델을 저장할 딕셔너리
# best_models1 = {}
# best_models2 = {}

# for modelname in benchmark_list:
#     max_val1 = 0.0
#     best_model1 = None
#     max_val2 = 0.0
#     best_model2 = None
#     # sav 파일 불러오기
#     for file in tqdm(os.listdir(PRJ_DIR), desc="files"):
#         if file.startswith(modelname):
#             with open(os.path.join(PRJ_DIR, file), 'rb') as f:
#                 loaded_model = pickle.load(f)

#             # 모델 성능 평가
#             tmp_pred = loaded_model.predict_proba(X_va)  # predict_prob 2d array
#             # print('predict_proba',tmp_pred)
#             tmp_pred = moving_average_segments(X_va_id, tmp_pred, window_size)
#             # print(tmp_pred)
#             max_pred = calculate_column_statistic(tmp_pred, 1, 'max')
#             average_pred = calculate_column_statistic(tmp_pred, 1, 'average')

#             # print(tmp_pred)
#             tmp_auroc1 = roc_auc_score(np.asarray(BN_Y_va), max_pred)
#             tmp_auroc2 = roc_auc_score(np.asarray(BN_Y_va), average_pred)

#             print(f'Max {modelname} | File: {file} | AUROC: {tmp_auroc1}')
#             print(f'Average {modelname} | File: {file} | AUROC: {tmp_auroc2}')

#             # 가장 높은 AUROC를 가진 모델 선택
#             if tmp_auroc1 > max_val1:
#                 max_val1 = tmp_auroc1
#                 best_model1 = loaded_model

#             # 가장 높은 AUROC를 가진 모델 선택
#             if tmp_auroc2 > max_val2:
#                 max_val2 = tmp_auroc2
#                 best_model2 = loaded_model

#     # 최종 선택된 베스트 모델 저장
#     if best_model1:
#         best_models1[modelname] = best_model1
#         filename = '{}/best_{}_w{}_{}.sav'.format(PRJ_DIR, modelname,window_size,'max')
#         with open(filename, 'wb') as f:
#             pickle.dump(best_model1, f)
#         print(f'Best max model for {modelname} saved with AUROC: {max_val1}')

#     # 최종 선택된 베스트 모델 저장
#     if best_model2:
#         best_models2[modelname] = best_model2
#         filename = '{}/best_{}_w{}_{}.sav'.format(PRJ_DIR, modelname,window_size,'average')
#         with open(filename, 'wb') as f:
#             pickle.dump(best_model2, f)
#         print(f'Best average model for {modelname} saved with AUROC: {max_val2}')


# available_benchmarks =['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest']
# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/benchmarks' --window_size 0 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/benchmarks' --window_size 3 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/benchmarks' --window_size 0 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/benchmarks' --window_size 3 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/benchmarks' --window_size 0 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest' --time_cut 90

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/benchmarks' --window_size 3 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest' --time_cut 90
################################################################################################################################################################
#

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2waystorage/benchmarks' --window_size 0 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2waystorage/benchmarks' --window_size 3 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'
################################################################################################################################################################
#

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_new_sc_bs_2waystorage/benchmarks' --window_size 0 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'

# CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python best_model_select.py --prj_dir '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_new_sc_bs_2waystorage/benchmarks' --window_size 3 --benchmarks 'Logistic Regression' 'XGBoost' 'Gradient Boosting' 'Random Forest'
