### == catholic
# last updated on 02/23/2024 calc_metric 추가
# 02/16/2024 : 인수인계로 인한 파일 경로 수정과 stage2 분석을 위한 데이터 path 수정
# 11/14/2023 : bootstrapping 
# 07/04/2023 : catholic full data ML
# 11/15/2022 : catholic sample data ML

# import libraries

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

#cpu affinity 
# cpu_set = 0
# num_cpu = 96
# p = psutil.Process()
# p.cpu_affinity(range(20,40))
# print(p.cpu_affinity())

import random
import sklearn



def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_dir', default='/storage/personal/myhwang/119/data/')  #for label 
    # parser.add_argument('--data_dir', default='/storage/personal/myhwang/NLP/data/data_assignments_2/total_tab_pain.csv')  #add pain scale label 
    parser.add_argument('--bow_dir', default='/storage/personal/myhwang/119/data/')  # for feature 
    parser.add_argument('--prj_dir', default='/storage/personal/myhwang/119/Topic_model/experiments_results/Topic_model/') #'/storage/personal/chkim/NLP/Topic_model/bow
    parser.add_argument('--config_dir', default='/storage/personal/myhwang/119/config/model_pipeline_config.yml')
    parser.add_argument('--save_dir', default=None, help='when need to save inference in the separate dir. eg ktas only test')
    parser.add_argument('--training', action='store_true', default=False, help='training or traning&inference')
    parser.add_argument('--scaling', action='store_true', default=False)
    parser.add_argument('--scaling_dir', action='store_true', default=False)
    parser.add_argument('--ktas_only', action='store_true', default=False, help='inference with ktas available data')
    parser.add_argument('--ktas_label', action='store_true', default=False, help='inference with ktas label')
    parser.add_argument('--bt', type=int, default=10, help="bootstrapping number")
    parser.add_argument('--sample_rate', type=float, default=0.75, help="bootstrapping sampling rate")
    parser.add_argument('--label', choices=['label', 'label1', 'label2'], default='label1') 
    parser.add_argument('--mode', choices=['2way', '3way'], default='2way')
    parser.add_argument('--window_size', type=int, default=0, help="window_size")
    parser.add_argument('--method', choices=['max', 'average'], default='max', help="method")
    parser.add_argument('--cpu_start', '-c', type=int, default=0, help="bootstrapping number")
    parser.add_argument('--time_cut', type=int, default=120, help="time cut for tf-idf")
    
    # label: 단순 심정지 여부
    # label1: 증상없음 -> 0, 심정지 or 호흡정지 -> 1, ‘심정지 or 호흡정지’가 없고 나머지 증상중 하나라도 있으면 -> 2
    # label2: ['흉통', '가슴불편감', '실신', '호흡곤란', '심계향진'] 중 하나라도 발생


    # parser.add_argument('--type', choices=['text', 'tabular', 'text_tabular'], default='text')
    # parser.add_argument('--single_stage', action='store_true', default=False)
    # parser.add_argument('--pain_score', action='store_true', default=False)
    return parser.parse_args()

# def get_ktas_df(df):
#     a = (~pd.isna(df["1ST_KTASLVLCD"])).values
#     b = (~pd.isna(df["2ND_KTASLVLCD"])).values
#     c = (~pd.isna(df["3RD_KTASLVLCD"])).values

#     ktas_df = df[a | b | c]

#     ktas_label_tag = {1:1, 2:1, 3:1, 4:0, 5:0}
#     ktas_level = []
#     ktas_label = []
#     ktas_temp = ktas_df[["1ST_KTASLVLCD", "2ND_KTASLVLCD", "3RD_KTASLVLCD"]]

#     for x, y, z in ktas_temp.values:
#         k = -1
#         if pd.isna(x):
#             if pd.isna(y):
#                 if pd.isna(z):
#                     k = -1
#                 else:
#                     k = z
#             else:
#                 k = y
#         else:
#             k = x
#         ktas_level.append(k)
#         ktas_label.append(ktas_label_tag[k])

#     ktas_df['KTAS_LV'] = ktas_level
#     ktas_df['KTAS_LABEL'] = ktas_label
#     return ktas_df

def main(args):
    

    #cpu affinity 
    p = psutil.Process()
    p.cpu_affinity(range(int(args.cpu_start),int(args.cpu_start)+5))
    print(p.cpu_affinity())
    print("now: ", now.strftime('%Y-%m-%d_%H:%M:%S'))
    PRJ_DIR = args.prj_dir
    

    # -- load configuration --
    
    CONFIG_PATH = args.config_dir
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    SEED = config['TRAIN']['seed']

    random.seed(SEED)
    np.random.seed(SEED)
    sklearn.random.seed(SEED)

    OUT_ITERATION = args.bt





    # -- Load Data --
    if args.time_cut ==120:
        tr_data = pd.read_csv(os.path.join(args.data_dir,'train_json_audio_data_decoder_time_arrest.csv'), index_col=0)
        va_data = pd.read_csv(os.path.join(args.data_dir,'valid_json_audio_data_decoder_time_arrest.csv'), index_col=0)
        te_data = pd.read_csv(os.path.join(args.data_dir,'test_json_audio_data_decoder_time_arrest.csv'), index_col=0)
    elif args.time_cut == 90:
        tr_data = pd.read_csv(os.path.join(args.data_dir,'train_json_audio_data_decoder_time_arrest.csv'), index_col=0)
        va_data = pd.read_csv(os.path.join('/home/myhwang/119_AMIGO/','valid_json_audio_data_decoder_time_arrest_90s.csv'), index_col=0)
        te_data = pd.read_csv(os.path.join('/home/myhwang/119_AMIGO/','test_json_audio_data_decoder_time_arrest_90s.csv'), index_col=0)
    elif args.time_cut == 60:
        tr_data = pd.read_csv(os.path.join(args.data_dir,'train_json_audio_data_decoder_time_arrest.csv'), index_col=0)
        va_data = pd.read_csv(os.path.join('/home/myhwang/119_AMIGO/','valid_json_audio_data_decoder_time_arrest_60s.csv'), index_col=0)
        te_data = pd.read_csv(os.path.join('/home/myhwang/119_AMIGO/','test_json_audio_data_decoder_time_arrest_60s.csv'), index_col=0)

    # -- Extract index --
    train_index = tr_data.index
    valid_index = va_data.index
    test_index = te_data.index

    # bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf.csv'), index_col='s_id') # 's_id'를 인덱스로 설정
    # tr_bow_df = bow_df.loc[train_index]
    # va_bow_df = bow_df[bow_df['id'].isin(valid_index)]
    # te_bow_df = bow_df[bow_df['id'].isin(test_index)]
    # -- Load Data --
    if args.time_cut ==120:
        tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train.csv'), index_col='id')
        va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf.csv'), index_col='s_id')
        te_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_test_tf_idf.csv'), index_col='s_id')
    elif args.time_cut == 90:
        tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train.csv'), index_col='id')
        va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf(tr)_90s.csv'), index_col='s_id')
        te_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_test_tf_idf(tr)_90s.csv'), index_col='s_id')
    elif args.time_cut == 60:
        tr_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','tf_idf_only_train.csv'), index_col='id')
        va_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_valid_tf_idf(tr)_60s.csv'), index_col='s_id')
        te_bow_df = pd.read_csv(os.path.join('/storage/personal/myhwang/119/data/','stacked_test_tf_idf(tr)_60s.csv'), index_col='s_id')

    # # # tr_bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf_only_train.csv'), index_col='id')
    # tr_bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf_only_train_mecab.csv'), index_col='id')
    # # va_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_valid_tf_idf_mecab.csv'), index_col='s_id')
    # # te_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_test_tf_idf_mecab.csv'), index_col='s_id')

    # va_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_valid_tf_idf(tr)_mecab.csv'), index_col='s_id')
    # te_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_test_tf_idf(tr)_mecab.csv'), index_col='s_id')

    # # tr_bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf_only_train_mecab_new.csv'), index_col='id')
    # # # va_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_valid_tf_idf_mecab_new.csv'), index_col='s_id')
    # # # te_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_test_tf_idf_mecab_new.csv'), index_col='s_id')

    # # va_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_valid_tf_idf(tr)_mecab_new.csv'), index_col='s_id')
    # # te_bow_df = pd.read_csv(os.path.join(args.bow_dir,'stacked_test_tf_idf(tr)_mecab_new.csv'), index_col='s_id')


    te_seq_index = te_bow_df.index
    # data = pd.read_csv(args.data_dir, index_col=0) # [07.04]label 1: 응급실 방문 권고 ,  나머지 0
    # bow_df = pd.read_csv(args.bow_dir, index_col=0)
    
    #label option 
    X_tr = tr_bow_df.loc[:, ~tr_bow_df.columns.isin(['Unnamed: 0', 'id'])]
    X_va = va_bow_df.loc[:, ~va_bow_df.columns.isin(['Unnamed: 0', 'id', 's_id'])]
    X_te = te_bow_df.loc[:, ~te_bow_df.columns.isin(['Unnamed: 0', 'id', 's_id'])]
    # X = bow_df.iloc[:,:-1]  # last column = sum
    X_va_id = va_bow_df.loc[:, ['id']]
    X_te_id = te_bow_df.loc[:, ['id']]

    Y_tr = tr_data.loc[:, args.label]
    Y_va = va_data.loc[:, args.label]
    Y_te = te_data.loc[:, args.label]

    Y_tr = Y_tr.apply(lambda x: 1 if x == 1 else 0) if args.mode == '2way' else Y_tr
    BN_Y_va = Y_va.apply(lambda x: 1 if x == 1 else 0)
    BN_Y_te = Y_te.apply(lambda x: 1 if x == 1 else 0)
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

    # -- Genereate Model Directory --
        
    if not os.path.exists(f'{PRJ_DIR}storage/benchmarks/'):
        os.makedirs(f'{PRJ_DIR}storage/benchmarks/')

    if not os.path.exists(f'{PRJ_DIR}storage/benchmarks/results/'):
        os.makedirs(f'{PRJ_DIR}storage/benchmarks/results/')

    if not os.path.exists(f'{args.save_dir}storage/results/'):
        os.makedirs(f'{args.save_dir}storage/results/')

    # -- Model Iteration --

    # benchmark_list = ['Random Forest'] 
    # benchmark_list = ['Gradient Boosting'] 
    # benchmark_list = ['XGBoost'] 
    # benchmark_list = ['Logistic Regression', 'XGBoost']
    benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest'] 
    
    # benchmark_list = ['Logistic Regression'] 
    # benchmark_list = ['XGBoost', 'Gradient Boosting', 'Random Forest'] 
    
    FINAL_RESULTS = np.zeros([OUT_ITERATION, len(benchmark_list), 10])

    # if args.single_stage:
    #     train_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/train_index_v2.npy')
    #     val_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/valid_index_v2.npy')
    #     test_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/test_index_v2.npy')
    # else:    
    #     train_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/train_index.npy')
    #     val_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/valid_index.npy')
    #     test_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/test_index.npy')
    
    # X_tr, X_t_tr, Y_tr = X.loc[train_index], X_tabular.loc[train_index], Y.loc[train_index]
    # X_va, X_t_va, Y_va = X.loc[val_index], X_tabular.loc[val_index], Y.loc[val_index]
    # print("train, val data shape", X_tr.shape, X_tr.shape, Y_tr.shape, X_va.shape, X_t_va.shape, Y_va.shape)
    print("train, val data shape", X_tr.shape, Y_tr.shape, X_va.shape, Y_va.shape)
    # Test will change depending on KTAS

    # --tree based model hyper parameter option
    set_max_depth = [1,2,3,4,5]
    max_depth_=0
    set_n_estimators = [100, 200, 300, 400, 500]
    n_estimators_=0
    C = np.logspace(-4, 4, 100)

    # selected_depth = 2
    # selected_n_estimator = 500

    results = np.zeros([len(benchmark_list), 10])

    benchmarks = {}
    
    # if args.ktas_only:
    #     print("Inference with KTAS available test data")
    #     train_df, test_df = data.loc[train_index], data.loc[test_index]  # need ktas info
    #     ktas_train, ktas_test = get_ktas_df(train_df), get_ktas_df(test_df)
    #     test_index = ktas_test.index
        
    # X_te, X_t_te, Y_te = X.loc[test_index], X_tabular.loc[test_index], Y.loc[test_index]
    # print("test data shape", X_te.shape, X_t_te.shape, Y_te.shape)

    # if args.ktas_label:  # replace bae label to ktas label , abort
    #     print("inference with KTAS label")
    #     Y_te = ktas_test[['KTAS_LABEL']]
        
    print("test data shape", X_te.shape, Y_te.shape)
    
    if args.scaling:
        if args.scaling_dir is False:  
            scaler = StandardScaler()
            scaled_X_tr = scaler.fit_transform(X_tr)
            #save scaler
            filename = '{}storage/benchmarks/scaler.sav'.format(PRJ_DIR)
            pickle.dump(scaler, open(filename, 'wb'))     
        else:  # use existing scaler
            filename = '{}storage/benchmarks/scaler.sav'.format(PRJ_DIR)
            with open (filename, 'rb') as f:
                scaler = pickle.load(f)
            print("scaler loaded")
        scaled_X_tr = pd.DataFrame(scaler.transform(X_tr), index=X_tr.index) 
        scaled_X_va = pd.DataFrame(scaler.transform(X_va), index=X_va.index)
        scaled_X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index)
        X_tr, X_va, X_te = scaled_X_tr.copy(), scaled_X_va.copy(), scaled_X_te.copy()
        # X_tr = scaled_X_tr.copy()
    # -- copy original data --
    
    label_frequency = float(Y_tr[Y_tr==1].sum() / len(Y_tr))

    
    # if args.ktas_label:
    #     Y_tr = ktas_train[['KTAS_LABEL']]
    #     label_frequency = float(Y_tr[Y_tr==1].sum() / len(Y_tr))

    print("label_frequency:", label_frequency)



    # if args.type == 'tabular':
    #     X_tr = X_t_tr
    #     X_va = X_t_va
    #     X_te = X_t_te
    #     X_tr.columns = X_tr.columns.astype(str)
    #     X_va.columns = X_va.columns.astype(str)
    #     X_te.columns = X_te.columns.astype(str)

    # elif args.type == 'text_tabular':
    #     X_tr = pd.merge(X_tr, X_t_tr ,left_index=True ,right_index=True ,how='inner')
    #     X_va = pd.merge(X_va, X_t_va ,left_index=True ,right_index=True ,how='inner')
    #     X_te = pd.merge(X_te, X_t_te ,left_index=True ,right_index=True ,how='inner')
    #     X_tr.columns = X_tr.columns.astype(str)
    #     X_va.columns = X_va.columns.astype(str)
    #     X_te.columns = X_te.columns.astype(str)


    if args.training is True:
        #Training
        print("Training")



        for m_idx, modelname in enumerate(benchmark_list):
            if modelname == 'Logistic Regression' :
                max_val = 0.
                for c in C:
                    tmp_model = LogisticRegression(random_state=SEED, C=c)
                    tmp_model.fit(X_tr, Y_tr)

                    tmp_pred = tmp_model.predict_proba(X_va)  # predict_prob 2d array 
                    
                    tmp_pred = moving_average_segments(X_va_id, tmp_pred, args.window_size)
                    tmp_pred = calculate_column_statistic(tmp_pred, 1, args.method)
                    tmp_auroc = roc_auc_score(np.asarray(BN_Y_va), tmp_pred)
                    
                    print('{}   | c:  {}   | AUROC: {}'.format(modelname, c, tmp_auroc))

                    if tmp_auroc > max_val : 
                        max_val = tmp_auroc
                        selected_c = c
                    #save model 
                    filename = '{}storage/benchmarks/{}_c{}.sav'.format(PRJ_DIR, modelname, c)
                    pickle.dump(tmp_model, open(filename, 'wb'))    
                    
                benchmarks[modelname] = LogisticRegression(random_state=SEED, C=selected_c)
                benchmarks[modelname].fit(X_tr, Y_tr) # 1d array
                

            elif modelname == 'XGBoost':
                max_val = 0.
                for max_depth_ in set_max_depth:
                    for n_estimators_ in set_n_estimators:
                        tmp_model = XGBClassifier(max_depth = max_depth_, n_estimators = n_estimators_, random_state =SEED)
                        tmp_model.fit(X_tr, Y_tr)

                        tmp_pred = tmp_model.predict_proba(X_va)  # predict_prob 2d array 

                        tmp_pred = moving_average_segments(X_va_id, tmp_pred, args.window_size)
                        tmp_pred = calculate_column_statistic(tmp_pred, 1, args.method)
                        tmp_auroc = roc_auc_score(np.asarray(BN_Y_va), tmp_pred)
                        

                        print('{}   | max_depth:  {}   | n_estimator:  {}    | AUROC: {}'.format(modelname, max_depth_, n_estimators_, tmp_auroc))

                        if tmp_auroc > max_val : 
                            max_val = tmp_auroc
                            selected_depth = max_depth_
                            selected_n_estimator = n_estimators_

                        #save model 
                        filename = '{}storage/benchmarks/{}_d{}_n{}.sav'.format(PRJ_DIR, modelname, max_depth_, n_estimators_)
                        pickle.dump(tmp_model, open(filename, 'wb'))    
                benchmarks[modelname] = XGBClassifier(max_depth=selected_depth, n_estimators=selected_n_estimator, random_state =SEED)
                benchmarks[modelname].fit(X_tr, Y_tr) #train with total train data 

            elif modelname == 'Gradient Boosting':
                max_val = 0.
                for max_depth_ in set_max_depth: 
                    for n_estimators_ in set_n_estimators:
                        tmp_model = GradientBoostingClassifier(max_depth = max_depth_, n_estimators = n_estimators_, random_state =SEED)
                        tmp_model.fit(X_tr, Y_tr)


                        tmp_pred = tmp_model.predict_proba(X_va)  # predict_prob 2d array 

                        tmp_pred = moving_average_segments(X_va_id, tmp_pred, args.window_size)
                        tmp_pred = calculate_column_statistic(tmp_pred, 1, args.method)
                        tmp_auroc = roc_auc_score(np.asarray(BN_Y_va), tmp_pred)
                        

                        print('{}     | max_depth:  {}  | n_estimator:  {}   | AUROC: {}'.format(modelname, max_depth_, n_estimators_, tmp_auroc))

                        if tmp_auroc > max_val:
                            max_val = tmp_auroc
                            selected_depth = max_depth_
                            selected_n_estimator = n_estimators_
                        #save model 
                        filename = '{}storage/benchmarks/{}_d{}_n{}.sav'.format(PRJ_DIR, modelname, max_depth_, n_estimators_)
                        pickle.dump(tmp_model, open(filename, 'wb'))

                benchmarks[modelname] = GradientBoostingClassifier(max_depth = selected_depth, n_estimators = selected_n_estimator, random_state =SEED)
                benchmarks[modelname].fit(X_tr, Y_tr)


            elif modelname == 'Random Forest':
                max_val = 0.
                for n_estimators_ in set_n_estimators:
                    # tmp_model = RandomForestClassifier(n_estimators=n_estimators_)
                    tmp_model = RandomForestClassifier(max_depth = max_depth_, n_estimators=n_estimators_, random_state =SEED)
                    # tmp_model = RandomForestClassifier(n_estimators=n_estimators_ ,class_weight='balanced')


                    tmp_model.fit(X_tr, Y_tr)


                    tmp_pred = tmp_model.predict_proba(X_va)  # predict_prob 2d array 

                    tmp_pred = moving_average_segments(X_va_id, tmp_pred, args.window_size)
                    tmp_pred = calculate_column_statistic(tmp_pred, 1, args.method)
                    tmp_auroc = roc_auc_score(np.asarray(BN_Y_va), tmp_pred)

                    print('{}  | max_depth: {}  | n_estimator: {}  | AUROC: {}'.format(modelname, max_depth_, n_estimators_, tmp_auroc))
                    if tmp_auroc > max_val:
                        max_val  = tmp_auroc
                        selected_n_estimator = n_estimators_
                    #save model 
                    filename = '{}storage/benchmarks/{}_d{}_n{}.sav'.format(PRJ_DIR, modelname, max_depth_, n_estimators_)
                    pickle.dump(tmp_model, open(filename, 'wb'))

                benchmarks[modelname] = RandomForestClassifier(max_depth = max_depth_, n_estimators=selected_n_estimator, random_state =SEED)
                # benchmarks[modelname] = RandomForestClassifier(n_estimators=selected_n_estimator ,class_weight='balanced')
                benchmarks[modelname].fit(X_tr, Y_tr)   
            
            #save model 
            filename = '{}storage/benchmarks/best_{}_w{}_{}.sav'.format(PRJ_DIR, modelname, args.window_size, args.method)
            pickle.dump(benchmarks[modelname], open(filename, 'wb'))            

    print("Inference")

    if args.save_dir is None:
        return print('Please pass save_dir for inference')
    #threshold label frequency [11.15]


    # # threshold Recall 기준 임의 설정
    if args.mode == '2way':
        if args.time_cut == 120:
            if args.window_size == 0:
                if args.method == 'max':
                    # threshold_dic={'Logistic Regression':0.06782, 'XGBoost':0.05521, 'Gradient Boosting':0.04259, 'Random Forest':0.02894} #mecab
                    threshold_dic={'Logistic Regression':0.0296, 'XGBoost':0.04822, 'Gradient Boosting':0.04259, 'Random Forest':0.02291} # mecab new
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.03231, 'Random Forest':0.04250}
            elif args.window_size == 3:
                if args.method == 'max':
                    # threshold_dic={'Logistic Regression':0.05861, 'XGBoost':0.05067, 'Gradient Boosting':0.03652, 'Random Forest':0.02727}
                    threshold_dic={'Logistic Regression':0.02979, 'XGBoost':0.04299, 'Gradient Boosting':0.031, 'Random Forest':0.02228} # mecab new
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
        elif args.time_cut == 90:
            if args.window_size == 0:
                if args.method == 'max':
                    # threshold_dic={'Logistic Regression':0.06782, 'XGBoost':0.05521, 'Gradient Boosting':0.04259, 'Random Forest':0.02894} #mecab
                    threshold_dic={'Logistic Regression':0.0296, 'XGBoost':0.04822, 'Gradient Boosting':0.04259, 'Random Forest':0.02291} # mecab new
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.03231, 'Random Forest':0.04250}
            elif args.window_size == 3:
                if args.method == 'max':
                    # threshold_dic={'Logistic Regression':0.05861, 'XGBoost':0.05067, 'Gradient Boosting':0.03652, 'Random Forest':0.02727}
                    threshold_dic={'Logistic Regression':0.02979, 'XGBoost':0.04299, 'Gradient Boosting':0.031, 'Random Forest':0.02228} # mecab new
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
        elif args.time_cut == 60:
            if args.window_size == 0:
                if args.method == 'max':
                    # threshold_dic={'Logistic Regression':0.06782, 'XGBoost':0.05521, 'Gradient Boosting':0.04259, 'Random Forest':0.02894} #mecab
                    threshold_dic={'Logistic Regression':0.0296, 'XGBoost':0.04822, 'Gradient Boosting':0.04259, 'Random Forest':0.02291} # mecab new
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.03231, 'Random Forest':0.04250}
            elif args.window_size == 3:
                if args.method == 'max':
                    # threshold_dic={'Logistic Regression':0.05861, 'XGBoost':0.05067, 'Gradient Boosting':0.03652, 'Random Forest':0.02727}
                    threshold_dic={'Logistic Regression':0.02979, 'XGBoost':0.04299, 'Gradient Boosting':0.031, 'Random Forest':0.02228} # mecab new
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
    
    elif args.mode == '3way':
        if args.time_cut == 120:
            if args.window_size == 0:
                if args.method == 'max':
                    threshold_dic={'Logistic Regression':0.02629, 'XGBoost':0.02351, 'Gradient Boosting':0.01879, 'Random Forest':0.01402} # mecab
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
            elif args.window_size == 3:
                if args.method == 'max':
                    threshold_dic={'Logistic Regression':0.03949, 'XGBoost':0.03656, 'Gradient Boosting':0.028, 'Random Forest':0.01881}
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.03949, 'XGBoost':0.03656, 'Gradient Boosting':0.028, 'Random Forest':0.01881}

        elif args.time_cut == 60:
            if args.window_size == 0:
                if args.method == 'max':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250} # mecab
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
            elif args.window_size == 3:
                if args.method == 'max':
                    threshold_dic={'Logistic Regression':0.0253, 'XGBoost':0.0210, 'Gradient Boosting':0.0173, 'Random Forest':0.0139}
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
        elif args.time_cut == 90:
            if args.window_size == 0:
                if args.method == 'max':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.02662, 'Gradient Boosting':0.02893, 'Random Forest':0.04250} # mecab
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
            elif args.window_size == 3:
                if args.method == 'max':
                    threshold_dic={'Logistic Regression':0.0272, 'XGBoost':0.0236, 'Gradient Boosting':0.0206, 'Random Forest':0.0147}
                elif args.method == 'average':
                    threshold_dic={'Logistic Regression':0.04424, 'XGBoost':0.04628, 'Gradient Boosting':0.02893, 'Random Forest':0.04250}
            

    print("Model output saving ")
    for m_idx, modelname in enumerate(benchmark_list):
        if modelname in threshold_dic.keys():
            threshold = threshold_dic[modelname]
        else:
            threshold = label_frequency 
        filename = f'{PRJ_DIR}storage/benchmarks/best_{modelname}_w{args.window_size}_{args.method}(train_idf)_{args.time_cut}.sav'#/storage/personal/myhwang/NLP/experiments_results/Topic_model/stage2/secondary_tf_idf_sc_storage/benchmarks
        with open(filename, 'rb') as f:
            benchmarks[modelname] = pickle.load(f)
        
        predicted_probas = benchmarks[modelname].predict_proba(X_te)
        seq_prob = moving_average_segments(X_te_id, predicted_probas, args.window_size)
        predicted_probas = calculate_column_statistic(seq_prob, 1, args.method)
        predicted_labels = np.where(predicted_probas >= threshold, 1, 0 )

        prediction_df = pd.DataFrame({'index': test_index,
                                        'predicted_probas': predicted_probas,
                                        'predicted_labels': predicted_labels,
                                        'label': BN_Y_te.squeeze()}, 
                                     index=test_index)
        prediction_df.to_csv('{}storage/results/{}_w{}_{}(tr_idf)_model_prediction_{}s_{}.csv'.format(args.save_dir, args.method, args.window_size, modelname, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')))
        # calc_metric_result_df = calc_metric(predicted_probas, Y_te.squeeze())
        calc_metric_result_df = calc_metric_roc_th(predicted_probas, BN_Y_te.squeeze())
        calc_metric_result_df.to_csv('{}storage/results/{}_w{}_{}(tr_idf)_calc_metric_results_{}s_{}.csv'.format(args.save_dir, args.method, args.window_size, modelname, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')), index=False)

        seq_prob_idx1 = np.concatenate([s[:, 1] for s in seq_prob]).flatten()
        seq_predicted_labels = np.where(seq_prob_idx1 >= threshold, 1, 0 )
        seq_prediction_df = pd.DataFrame({  'id' : np.array(X_te_id).flatten(),
                                            'predicted_probas': seq_prob_idx1,
                                            'seq_predicted_labels': seq_predicted_labels}
                                            ,index=te_seq_index)
        result_df = seq_prediction_df.merge(BN_Y_te, left_on='id', right_index=True, how='inner')
        result_df.to_csv('{}storage/results/{}_w{}_{}(tr_idf)_model_seq_prediction_{}s_{}.csv'.format(args.save_dir, args.method, args.window_size, modelname, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')))



    # 유일한 id 값을 추출
    ids = X_te_id['id'].unique()

    for out_itr in range(OUT_ITERATION):  # Bootstrapping 반복
        print(f"{out_itr} / {OUT_ITERATION}")
        
        # RandomState 객체를 사용하여 재현 가능한 샘플링 설정
        rng = np.random.RandomState(SEED + out_itr)
        
        # ids 리스트에서 args.sample_rate 비율만큼 샘플링
        sampled_ids = rng.choice(ids, size=int(len(ids) * args.sample_rate), replace=True)
        # 샘플링된 id와 같은 id를 가진 행만 추출
        X_te_ = X_te[X_te_id['id'].isin(sampled_ids)]
        Y_te_ = BN_Y_te[BN_Y_te.index.isin(sampled_ids)]
        X_te_id_ = X_te_id[X_te_id['id'].isin(sampled_ids)]
        

        inference_df = te_data.loc[sampled_ids,['text','label1']]
        inference_df['label1'] = inference_df['label1'].apply(lambda x: 1 if x == 1 else 0)

        inference_df.columns = ['data', 'label']
        inference_df.to_csv('{}storage/results/inference(tr_idf)_itr{}_{}s_{}.csv'.format(args.save_dir, out_itr, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')))

        for m_idx, modelname in enumerate(benchmark_list):
            filename = f'{PRJ_DIR}storage/benchmarks/best_{modelname}_w{args.window_size}_{args.method}(train_idf)_{args.time_cut}.sav'#/storage/personal/myhwang/NLP/experiments_results/Topic_model/stage2/secondary_tf_idf_sc_storage/benchmarks
            # filename = f'{PRJ_DIR}storage/benchmarks/best_{modelname}_w{args.window_size}_{args.method}(train_idf)_{args.time_cut}.sav'#/storage/personal/myhwang/NLP/experiments_results/Topic_model/stage2/secondary_tf_idf_sc_storage/benchmarks
            with open(filename, 'rb') as f:
                benchmarks[modelname] = pickle.load(f)
            # benchmarks[modelname] = pickle.load(open(filename, 'rb'))
            # # # threshold Recall 기준 임의 설정
            threshold=threshold_dic[f'{modelname}']
            
            # threshold = label_frequency # threshol를 label_frequency로 설정
            
            pred        = benchmarks[modelname].predict_proba(X_te_)
            pred        = moving_average_segments(X_te_id_, pred, args.window_size)
            pred        = calculate_column_statistic(pred, 1, args.method)

            auroc       = roc_auc_score(Y_te_, pred)
            auprc       = average_precision_score(Y_te_, pred)


            tmp_pred_ = np.where(pred >= threshold, 1, 0 )
            
            f1          = f1_score(Y_te_, tmp_pred_)
            acc         = accuracy_score(Y_te_, tmp_pred_)
            brier = brier_score_loss(Y_te_,pred) # proba
            precision = precision_score(Y_te_,tmp_pred_)
            recall = recall_score(Y_te_,tmp_pred_)
            specificity = specificity_score(Y_te_,tmp_pred_)
            tn, fp, fn, tp = confusion_matrix(Y_te_, tmp_pred_).ravel()
            
            # NPV 계산 (분모가 0이 되는 경우 방지)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

            results[m_idx, 0] = auroc
            results[m_idx, 1] = auprc
            results[m_idx, 2] = f1
            results[m_idx, 3] = acc
            results[m_idx, 4] = brier
            results[m_idx, 5] = recall
            results[m_idx, 6] = precision
            results[m_idx, 7] = specificity
            results[m_idx, 8] = npv
            results[m_idx, 9] = fdr
        
        if args.save_dir is not None: 
            print(f'saving in {args.save_dir}')
            pd.DataFrame(results, index=benchmark_list, columns=['AUROC', 'AUPRC', 'F1', 'Accuracy', 'Brier', 'Recall', 'Precision', 'Specificity', 'NPV', 'FDR']).to_csv('{}storage/results/tmp_results_ML(tr_idf)_itr{}_{}s_{}.csv'.format(args.save_dir, out_itr, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')))
        else:   
            pd.DataFrame(results, index=benchmark_list, columns=['AUROC', 'AUPRC', 'F1', 'Accuracy', 'Brier', 'Recall', 'Precision', 'Specificity', 'NPV', 'FDR']).to_csv('{}storage/benchmarks/results/tmp_results_ML(tr_idf)_itr{}_{}s_{}.csv'.format(PRJ_DIR, out_itr, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')))

        FINAL_RESULTS[out_itr, :, :] = results

    final_result_df = pd.DataFrame(
                    np.concatenate([FINAL_RESULTS.mean(axis=0), FINAL_RESULTS.std(axis=0)], axis=1), 
                    index=benchmark_list, 
                    columns=['AUROC_mean', 'AUPRC_mean', 'F1_mean' ,'Accuracy_mean', 'Brier_mean', 'Recall_mean', 'Precision_mean', 'Specificity_mean', 'NPV_mean', 'FDR_mean'] + ['AUROC_std', 'AUPRC_std', 'F1_std', 'Accuracy_std', 'Brier_std', 'Recall_std', 'Precision_std', 'Specificity_std', 'NPV_std', 'FDR_std'])

    if args.save_dir is None:
        final_result_df.to_csv('{}storage/benchmarks/results/results_ML(tr_idf)_w{}_{}_{}s_{}.csv'.format(PRJ_DIR, args.window_size, args.method, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')))
        
    else:
        print(f'saving in {args.save_dir}')
        final_result_df.to_csv('{}storage/results/results_ML(tr_idf)_w{}_{}_{}s_{}.csv'.format(args.save_dir, args.window_size, args.method, args.time_cut, now.strftime('%Y-%m-%d_%H:%M:%S')))
    
    print("export final result df")


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

if __name__ == '__main__':
    args = parse_args()
    main(args)    



#for train

#2way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 0 --method 'max'

#2way window 0 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 0 --method 'average'

#2way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 3 --method 'max'

#2way window 3 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 3 --method 'average'

#3way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --training --label 'label1' --mode '3way' --window_size 0 --method 'max'

#3way window 0 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --training --label 'label1' --mode '3way' --window_size 0 --method 'average'

#3way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --training --label 'label1' --mode '3way' --window_size 3 --method 'max'

#3way window 3 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --training --label 'label1' --mode '3way' --window_size 3 --method 'average'

##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
#for for evaluation

#2way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --label 'label1' --mode '2way' --window_size 0 --method 'max'

#2way window 0 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --label 'label1' --mode '2way' --window_size 0 --method 'average'

#2way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --label 'label1' --mode '2way' --window_size 3 --method 'max'

#2way window 3 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --label 'label1' --mode '2way' --window_size 3 --method 'average'

#3way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 0 --method 'max'

#3way window 0 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 0 --method 'average'

#3way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 3 --method 'max'

#3way window 3 average
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 3 --method 'average'


#3way window 0 max 90s
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 0 --method 'max' --time_cut 90

#3way window 0 average 90s
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 0 --method 'average' --time_cut 90

#3way window 3 max 90s
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 3 --method 'max' --time_cut 90

#3way window 3 average 90s
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --label 'label1' --mode '3way' --window_size 3 --method 'average' --time_cut 90


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#for train

#2way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 0 --method 'max'

#2way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 3 --method 'max'

#for for evaluation

#2way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --label 'label1' --mode '2way' --window_size 0 --method 'max'

#2way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --label 'label1' --mode '2way' --window_size 3 --method 'max'
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#for train

#2way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 0 --method 'max'

#2way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_sc_bs_2way --training --label 'label1' --mode '2way' --window_size 3 --method 'max'

#for for evaluation

#2way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_new_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_new_sc_bs_2way --label 'label1' --mode '2way' --window_size 0 --method 'max'

#2way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python main_bootstrap_arrest_time.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_new_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_mecab_new_sc_bs_2way --label 'label1' --mode '2way' --window_size 3 --method 'max'
