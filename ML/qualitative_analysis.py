# Qualitative analysis 
"""
AUROC curve & Precision recall curve for models 
"""
# updated on 2024/1/23 중증 directory

## === import libraries
import numpy as np
import pandas as pd
import os, sys
import pickle
import yaml
import pdb

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, brier_score_loss, f1_score, accuracy_score, precision_recall_curve

# libraries for benchmark models

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

import joblib
import psutil
import argparse
import matplotlib.pyplot as plt

import random
import sklearn

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', default='/storage/personal/myhwang/NLP/data/data_assignments_2/total_tab.csv')  #for label 
#     parser.add_argument('--bow_dir', default='/storage/personal/myhwang/NLP/Topic_model/bag_of_words/tf_idf.csv')  # for feature 
#     parser.add_argument('--prj_dir', default='/storage/personal/myhwang/NLP/experiments_results/Topic_model/')
#     parser.add_argument('--config_dir', default='/storage/personal/myhwang/NLP/Topic_model/config/model_pipeline_config.yml')
#     parser.add_argument('--save_dir', default=None, help='when need to save inference in the separate dir. eg ktas only test')
#     parser.add_argument('--scaling', action='store_true', default=False)
#     parser.add_argument('--ktas_only', action='store_true', default=False, help='inference with ktas available data')
#     parser.add_argument('--ktas_label', action='store_true', default=False, help='inference with ktas label')
#     parser.add_argument('--label', choices=['primary_outcome', 'secondary_outcome'], default='primary_outcome')
#     parser.add_argument('--type', choices=['text', 'tabular', 'text_tabular'], default='text')
#     parser.add_argument('--single_stage', action='store_true', default=False)
#     return parser.parse_args()
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_dir', default='/storage/personal/myhwang/119/data/')  #for label 
    # parser.add_argument('--data_dir', default='/storage/personal/myhwang/NLP/data/data_assignments_2/total_tab_pain.csv')  #add pain scale label 
    # parser.add_argument('--bow_dir', default='/storage/personal/myhwang/119/data/')  # for feature 
    parser.add_argument('--prj_dir', default='/storage/personal/myhwang/119/Topic_model/experiments_results/Topic_model/') #'/storage/personal/chkim/NLP/Topic_model/bow
    parser.add_argument('--config_dir', default='/storage/personal/myhwang/119/config/model_pipeline_config.yml')
    parser.add_argument('--save_dir', default=None, help='when need to save inference in the separate dir. eg ktas only test')
    parser.add_argument('--scaling', action='store_true', default=False)
    parser.add_argument('--scaling_dir', action='store_true', default=False)
    parser.add_argument('--ktas_only', action='store_true', default=False, help='inference with ktas available data')
    parser.add_argument('--ktas_label', action='store_true', default=False, help='inference with ktas label')
    # parser.add_argument('--label', choices=['label', 'label1', 'label2'], default='label1')
    parser.add_argument('--mode', choices=['2way', '3way'], default='2way')
    parser.add_argument('--cpu_start', '-c', type=int, default=0, help="bootstrapping number")
    parser.add_argument('--window_size', type=int, default=0, help="window_size")
    parser.add_argument('--method', choices=['max', 'average'], default='max', help="method")
    parser.add_argument('--time_stamp', type=str, help="result timestamp")
    parser.add_argument('--time_cut', type=int, default=120, help='time_cut')
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
    p.cpu_affinity(range(int(args.cpu_start),int(args.cpu_start)+10))
    print(p.cpu_affinity())

    PRJ_DIR = args.prj_dir
    # -- load configuration --

    CONFIG_PATH = args.config_dir
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    if args.save_dir:
        if not os.path.exists(f'{args.save_dir}storage/quantitiative_result/'):
            os.makedirs(f'{args.save_dir}storage/quantitiative_result/')


    sys.path.append(PRJ_DIR)

    SEED = config['TRAIN']['seed']

    OUT_ITERATION = config['TRAIN']['out_iteration']


    random.seed(SEED)
    np.random.seed(SEED)
    sklearn.random.seed(SEED)

    # # -- Load Data --
    # tr_data = pd.read_csv(os.path.join(args.data_dir,'train_json_audio_data_decoder_time_arrest.csv'), index_col=0)
    # va_data = pd.read_csv(os.path.join(args.data_dir,'valid_json_audio_data_decoder_time_arrest.csv'), index_col=0)
    # te_data = pd.read_csv(os.path.join(args.data_dir,'test_json_audio_data_decoder_time_arrest.csv'), index_col=0)

    # # -- Extract index --
    # train_index = tr_data.index
    # valid_index = va_data.index
    # test_index = te_data.index

    # # bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf.csv'), index_col=-1)
    # # tr_bow_df = bow_df.loc[train_index]
    # # va_bow_df = bow_df.loc[valid_index]
    # # te_bow_df = bow_df.loc[test_index]

    # bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf.csv'), index_col=-1)
    # tr_bow_df = pd.read_csv(os.path.join(args.bow_dir,'tf_idf_only_train.csv'), index_col=-1)
    # va_bow_df = bow_df.loc[valid_index]
    # te_bow_df = bow_df.loc[test_index]
    # # # -- Load Data --
    # # tr_bow_df = pd.read_csv(os.path.join(args.bow_dir,'train_tf_idf.csv'), index_col=0)
    # # vl_bow_df = pd.read_csv(os.path.join(args.bow_dir,'valid_tf_idf.csv'), index_col=0)
    # # te_bow_df = pd.read_csv(os.path.join(args.bow_dir,'test_tf_idf.csv'), index_col=0)

    # # data = pd.read_csv(args.data_dir, index_col=0) # [07.04]label 1: 응급실 방문 권고 ,  나머지 0
    # # bow_df = pd.read_csv(args.bow_dir, index_col=0)
    
    # #label option 
    # X_tr = tr_bow_df.loc[:, ~tr_bow_df.columns.isin(['Unnamed: 0', 'id'])]
    # X_va = va_bow_df.loc[:, ~va_bow_df.columns.isin(['Unnamed: 0', 'id'])]
    # X_te = te_bow_df.loc[:, ~te_bow_df.columns.isin(['Unnamed: 0', 'id'])]
    # # X = bow_df.iloc[:,:-1]  # last column = sum

    # Y_tr = tr_data.loc[:, args.label]
    # Y_va = va_data.loc[:, args.label]
    # Y_te = te_data.loc[:, args.label]

    # Y_tr = Y_tr.apply(lambda x: 1 if x == 1 else 0) if args.mode == '2way' else Y_tr
    # BN_Y_va = Y_va.apply(lambda x: 1 if x == 1 else 0)
    # BN_Y_te = Y_te.apply(lambda x: 1 if x == 1 else 0)

    # # model 학습에서와 같은 seed 고정으로 stratified kfold 
        
    # skf = StratifiedKFold(n_splits = OUT_ITERATION, random_state = SEED, shuffle = True)

    ### benchmark list
    # benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest', 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM'] 
    # benchmark_list = ['XGBoost', 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM'] # Primary Text Logistic Regression
    # benchmark_list = ['XGBoost', 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM'] # Primary Text + Tabular Logistic Regression
    # benchmark_list = ['XGBoost', 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM'] # Secondary Text XGBoost 
    # benchmark_list = ['XGBoost', 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM'] # Secondary Text + Tabular XGBoost
    # benchmark_list = ['Random Forest', 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM']
    # benchmark_list = ['Logistic Regression', 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM']
    # benchmark_list = ['KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM']
    # benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest']
    # benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest', 'Tabular_Only_DL']  
    # benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest', 'KM_BERT_B', 'KM_BERT_B_MLM'] 
    # benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest', 'Our (w/o mv_Avg_3)', 'Our (w mv_Avg_3)']  
    benchmark_list = ['Logistic Regression', 'XGBoost', 'Gradient Boosting', 'Random Forest', 'DyLM-OHCA']  
    # benchmark_list = ['Tabular_Only_DL']  
    # create directory

    for method in benchmark_list:
        if not os.path.exists(f'{PRJ_DIR}storage/quantitiative_result/'):
            os.makedirs(f'{PRJ_DIR}storage/quantitiative_result/')
        
    ##=====AUROC CURVE



    benchmarks = {}
    result = {}

    # if args.single_stage:
    #     train_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/train_index_v2.npy')
    #     val_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/valid_index_v2.npy')
    #     test_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/test_index_v2.npy')
    # else:    
    #     train_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/train_index.npy')
    #     val_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/valid_index.npy')
    #     test_index = np.load('/storage/personal/myhwang/NLP/data/data_assignments_2/test_index.npy')


    
    # if args.ktas_only:
    #     print("Inference with KTAS available ")
    #     train_df, test_df = data.iloc[train_index,:], data.iloc[test_index,:]
    #     ktas_train, ktas_test = get_ktas_df(train_df), get_ktas_df(test_df)
    #     test_index = ktas_test.index
    
    # X_tr, X_te = X.loc[train_index, :], X.loc[test_index, :]
    # X_t_tr, X_t_te = X_tabular.loc[train_index, :], X_tabular.loc[test_index, :]
    # Y_tr, Y_te = Y.loc[train_index], Y.loc[test_index]

    # if args.ktas_label:  # replace bae label to ktas label 
    #     print("inference with KTAS label")
    #     Y_te = ktas_test[['KTAS_LABEL']]

    # X_tr = X_tr.reset_index(drop=True)
    # X_te = X_te.reset_index(drop=True)
    # Y_tr = Y_tr.reset_index(drop=True)
    # Y_te = Y_te.reset_index(drop=True)
    
    
    # if args.scaling:
    #     filename = '{}storage/benchmarks/scaler.sav'.format(PRJ_DIR)
    #     with open (filename, 'rb') as f:
    #         scaler = pickle.load(f)
    #     scaled_X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index)
    #     X_te = scaled_X_te.copy()
        
    # if args.type == 'tabular':
    #     X_te = X_t_te
    #     X_te.columns = X_te.columns.astype(str)

    # elif args.type == 'text_tabular':
    #     X_te = pd.merge(X_te, X_t_te ,left_index=True ,right_index=True ,how='inner')
    #     X_te.columns = X_te.columns.astype(str)

    for method in benchmark_list:
        
        result[method] =[]

        if method == 'Our (w/o mv_Avg_3)':  # get predicted label 

            filename = '/storage/personal/myhwang/119/DL/result/np_60s/inference_logit_checkpoint_8_710_None_0.csv'

            # filename = '{}storage/{}_{}_model_prediction_{}.csv'.format(PRJ_DIR, args.label, method, args.type)
            y = pd.read_csv(filename, index_col=0)
            Y_te = y['labels']
            # y_pred = y['predicted_labels']
            y_proba = y['predicted_probas']
            fpr, tpr, thresholds = roc_curve(Y_te, y_proba)
            precision, recall, _ = precision_recall_curve(Y_te, y_proba)

        elif method == 'DyLM-OHCA':  # get predicted label 

            filename = '/storage/personal/myhwang/119/DL/result/moving_Avg3_60s/inference_logit_checkpoint_8_710_moving_average_3.csv'
            # filename = '{}storage/{}_{}_model_prediction_{}.csv'.format(PRJ_DIR, args.label, method, args.type)
            y = pd.read_csv(filename, index_col=0)
            Y_te = y['labels']
            # y_pred = y['predicted_labels']
            y_proba = y['predicted_probas']
            fpr, tpr, thresholds = roc_curve(Y_te, y_proba)
            precision, recall, _ = precision_recall_curve(Y_te, y_proba)
        else:
            filename = f'{PRJ_DIR}storage/results/{args.method}_w{args.window_size}_{method}(tr_idf)_model_prediction_{args.time_cut}s_{args.time_stamp}.csv'
            y = pd.read_csv(filename, index_col=0)
            Y_te = y['label']
            # y_pred = y['predicted_labels']
            y_proba = y['predicted_probas']
            fpr, tpr, thresholds = roc_curve(Y_te, y_proba)
            precision, recall, _ = precision_recall_curve(Y_te, y_proba)


        # if method in [ 'KM_BERT_S', 'KM_BERT_S_MLM', 'KM_BERT_B', 'KM_BERT_B_MLM', 'Tabular_Only_DL', 'KM-BERT', 'KM-BERT with MLM']:  # get predicted label 
        #     filename = '{}storage/{}_{}_model_prediction_{}.csv'.format(PRJ_DIR, args.label, method, args.type)
        #     y = pd.read_csv(filename, index_col=0)
        #     Y_te = y['label']
        #     y_pred = y['predicted_labels']
        #     y_proba = y['predicted_probas']
        #     fpr, tpr, thresholds = roc_curve(Y_te, y_proba)
        #     precision, recall, _ = precision_recall_curve(Y_te, y_proba)
        # else:
        #     filename = f'{PRJ_DIR}storage/benchmarks/{method}_{args.type}.sav'
        #     model = pickle.load(open(filename, 'rb'))
                
        #     y_proba = model.predict_proba(X_te)
        #     fpr, tpr, thresholds = roc_curve(Y_te, y_proba[:,1])
        #     precision, recall, _ = precision_recall_curve(Y_te, y_proba[:,1])
        #brier = brier_score_loss(Y_te, y_pred[:,1])
        
        result[method].append(fpr)
        result[method].append(tpr)
        result[method].append(thresholds)
        result[method].append(precision)
        result[method].append(recall)
        #result[method].append(brier)

    no_skill = np.sum(Y_te == 1)/len(Y_te)
    
    print(f"no_skill threshold:{no_skill}")
    y_pred_ns  = [0 for _ in range(len(Y_te))] #null model
    fpr_ns, tpr_ns, thresholds_ns  = roc_curve(Y_te, y_pred_ns)
    
    ##AUROC for models 

    plt.figure(figsize=[6,6])
    # plt.plot(fpr_ns, tpr_ns, linestyle='--', linewidth=1.5, label='No Skill', color='black')
    plt.plot(fpr_ns, tpr_ns, linestyle='--', linewidth=1.5, color='black')
    
    for model, value in result.items():
        plt.plot(value[0], value[1], linestyle='--', linewidth=2.0, label=model)
    
    # axis labels
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0.,1.])
    plt.xlim([0.,1.])

    # show the legend
    # plt.grid(color='gray', linestyle='-', alpha=0.5)
    plt.legend(loc = 'lower right', fontsize=13)
    # plt.title(f"{args.label} AUROC", fontsize = 16)
    
    if args.save_dir is not None:
        print(f'saving in {args.save_dir}')
        plt.savefig(f'{args.save_dir}storage/quantitiative_result/auroc_curve_{args.mode}_w{args.window_size}_{args.time_cut}s.png', dpi=300)
    else:
        plt.savefig(f'{PRJ_DIR}storage/quantitiative_result/auroc_curve_{args.mode}_w{args.window_size}_{args.time_cut}s.png', dpi=300)
    plt.show()
    plt.close()
        
    #Precision Recall Curve for models 

    plt.figure(figsize=(6,6))
    # plt.plot([0,1], [no_skill, no_skill], linestyle = '--', linewidth = 1.5, label='No Skill', color = 'black')
    plt.plot([0,1], [no_skill, no_skill], linestyle = '--', linewidth = 1.5, color = 'black')

    for model, value in result.items():
        plt.plot(value[4], value[3], linestyle = '--', linewidth=2.0, label=model)

    #axis
    plt.xlabel('Recall', fontsize = 16)
    plt.ylabel('Precision', fontsize = 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([-0.00, 1.00])
    plt.xlim([-0.00,1.00])
        #show the legned
    # plt.grid(color='gray', linestyle='-', alpha=0.5)
    plt.legend(loc='upper right', fontsize=13)

    # plt.title(f'{args.label} Precision-Recall', fontsize = 16)
    if args.save_dir is not None:
        print(f'saving in {args.save_dir}')
        plt.savefig(f'{args.save_dir}storage/quantitiative_result/precision_recall_curve_{args.mode}_w{args.window_size}_{args.time_cut}s.png', dpi=300)
    else:
        plt.savefig(f'{PRJ_DIR}storage/quantitiative_result/precision_recall_curve_{args.mode}_w{args.window_size}_{args.time_cut}s.png', dpi=300)
    plt.show()
    plt.close()
    print("Done Saving")

if __name__ == '__main__':
    args = parse_args()
    main(args)

#2way window 0 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python qualitative_analysis.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --mode '2way' --window_size 0 --method 'max' --time_stamp "2024-08-26_22:18:29"

#2way window 3 max
#CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=10 python qualitative_analysis.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2way --mode '2way' --window_size 3 --method 'max' --time_stamp "2024-08-27_11:02:46"

# OMP_NUM_THREADS=10 python qualitative_analysis.py --prj_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --scaling --save_dir /storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3way --mode '3way' --window_size 3 --method 'max' --time_stamp "2025-08-27_16:37:51" --time_cut 60