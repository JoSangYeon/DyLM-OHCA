import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 원본 함수
def text_mapping_stack(df, predict_df):
    predict_df['starttime'] = None
    predict_df['endtime'] = None  
    id_ = ''
    for t in tqdm(range(len(predict_df))):
        if id_ != predict_df['id'].iloc[t]:
            i = 0
            id_ = predict_df['id'].iloc[t]
        
        try:
            predict_df.loc[t, 'starttime'] = int(df[df['id'] == id_].iloc[:, 10 + i])
            predict_df.loc[t, 'endtime'] = int(df[df['id'] == id_].iloc[:, 11 + i])
        except IndexError:
            print(f"IndexError at t={t}, id={id_}, i={i}")
        
        i = i + 4

# 병렬로 실행할 작업 정의
def process_file(df, file_path, output_path):
    predict_df = pd.read_csv(file_path)
    text_mapping_stack(df, predict_df)
    predict_df.to_csv(output_path, index=False)

# # 병렬로 처리할 파일 목록
# file_paths = [
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Gradient Boosting_model_seq_prediction_2024-08-26_22:18:29.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Gradient Boosting_model_seq_prediction_2024-08-26_22:18:29_time.csv"),
    
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Logistic Regression_model_seq_prediction_2024-08-26_22:18:29.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Logistic Regression_model_seq_prediction_2024-08-26_22:18:29_time.csv"),
    
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Random Forest_model_seq_prediction_2024-08-26_22:18:29.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Random Forest_model_seq_prediction_2024-08-26_22:18:29_time.csv"),
    
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_XGBoost_model_seq_prediction_2024-08-26_22:18:29.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_XGBoost_model_seq_prediction_2024-08-26_22:18:29_time.csv")
# ]
# 병렬로 처리할 파일 목록
# file_paths = [
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Gradient Boosting_model_seq_prediction_2024-08-27_11:02:46.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Gradient Boosting_model_seq_prediction_2024-08-27_11:02:46_time.csv"),
    
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Logistic Regression_model_seq_prediction_2024-08-27_11:02:46.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Logistic Regression_model_seq_prediction_2024-08-27_11:02:46_time.csv"),
    
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Random Forest_model_seq_prediction_2024-08-27_11:02:46.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Random Forest_model_seq_prediction_2024-08-27_11:02:46_time.csv"),
    
#     ("/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_XGBoost_model_seq_prediction_2024-08-27_11:02:46.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_XGBoost_model_seq_prediction_2024-08-27_11:02:46_time.csv")
# ]
# file_paths = [
#     ("/mnt/storage/personal/myhwang/119/experiments_results/inference_file_checkpoint_8_710_moving_average_3.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/inference_file_checkpoint_8_710_moving_average_3_time.csv"),
    
#     ("/mnt/storage/personal/myhwang/119/experiments_results/inference_file_checkpoint_8_710_None_3.csv", 
#      "/mnt/storage/personal/myhwang/119/experiments_results/inference_file_checkpoint_8_710_None_3_time.csv"),
# ]

file_paths = [
    # ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51.csv",
    # "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv",),

    # ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51.csv",
    # "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv",),

    ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51.csv",
    "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv",),

    # ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51.csv",
    # "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv",),


    # ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56.csv",
    # "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv",),

    # ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56.csv",
    # "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv",),

    ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56.csv",
    "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv",),

    ("/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56.csv",
    "/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv",),

]

# 원본 데이터 읽기
df = pd.read_csv('/storage/personal/myhwang/119/data/test_json_audio_data_decoder_time_arrest.csv')

# ThreadPoolExecutor를 사용하여 병렬 처리
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, df, file_path, output_path) for file_path, output_path in file_paths]
    for future in as_completed(futures):
        try:
            future.result()  # 작업의 완료를 기다림
        except Exception as e:
            print(f"Error occurred: {e}")

print("All files processed.")
