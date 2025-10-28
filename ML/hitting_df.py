import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def process_id(group):
    id_, group = group
    # predicted_labels 설정
    if (group['seq_predicted_labels'] == 1).any():
        predicted_label = 1
    else:
        predicted_label = 0

    # TP, FN, FP, TN 계산
    label1 = group['label1'].iloc[0]  # 모든 row에서 label1은 동일하다고 가정
    if predicted_label == 1 and label1 == 1:
        category = 'TP'
    elif predicted_label == 0 and label1 == 1:
        category = 'FN'
    elif predicted_label == 1 and label1 == 0:
        category = 'FP'
    else:
        category = 'TN'

    # 처음 1이 되는 지점의 starttime, endtime, s_id 찾기
    if predicted_label == 1:
        first_one_index = group[group['seq_predicted_labels'] == 1].index[0]
        starttime = group.loc[first_one_index, 'starttime']
        endtime = group.loc[first_one_index, 'endtime']
        s_num = group.loc[first_one_index, 's_num']
    else:
        # 1이 없을 경우, 마지막 row의 starttime, endtime, s_id 사용
        starttime = group['starttime'].iloc[-1]
        endtime = group['endtime'].iloc[-1]
        s_num = -1

    # 결과 반환
    return {
        'id': id_,
        's_num': int(s_num),
        'predicted_labels': predicted_label,
        'label1': label1,
        'category': category,
        'starttime': starttime,
        'endtime': endtime,
    }

def process_file(input_file, output_file):
    # 데이터 로드
    df = pd.read_csv(input_file)
    df['s_num'] = df.groupby('id').cumcount()+1

    # id별로 그룹화하여 병렬 처리
    grouped = df.groupby('id')
    # ProcessPoolExecutor를 사용하여 병렬 처리
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_id, group) for group in grouped]
        results = [future.result() for future in as_completed(futures)]

    # 결과를 데이터프레임으로 변환 및 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

# 병렬로 처리할 파일 리스트 정의
files_to_process = [
    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Gradient Boosting_model_seq_prediction_2024-08-26_22:18:29_time.csv',
    #  'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Gradient Boosting_model_seq_prediction_2024-08-26_22:18:29_time_hit.csv'},
    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Logistic Regression_model_seq_prediction_2024-08-26_22:18:29_time.csv'
    # , 'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Logistic Regression_model_seq_prediction_2024-08-26_22:18:29_time_hit.csv'},
    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Random Forest_model_seq_prediction_2024-08-26_22:18:29_time.csv'
    # , 'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_Random Forest_model_seq_prediction_2024-08-26_22:18:2_time_hit.csv'},
    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_XGBoost_model_seq_prediction_2024-08-26_22:18:29_time.csv'
    # , 'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w0_XGBoost_model_seq_prediction_2024-08-26_22:18:29_time_hit.csv'},

    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Gradient Boosting_model_seq_prediction_2024-08-27_11:02:46_time.csv'
    # , 'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Gradient Boosting_model_seq_prediction_2024-08-27_11:02:46_time_hit.csv'},
    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Logistic Regression_model_seq_prediction_2024-08-27_11:02:46_time.csv'
    # , 'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Logistic Regression_model_seq_prediction_2024-08-27_11:02:46_time_hit.csv'},
    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Random Forest_model_seq_prediction_2024-08-27_11:02:46_time.csv'
    # , 'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_Random Forest_model_seq_prediction_2024-08-27_11:02:46_time_hit.csv'},
    # {'input_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_XGBoost_model_seq_prediction_2024-08-27_11:02:46_time.csv'
    # , 'output_file': '/mnt/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_2waystorage/results/max_w3_XGBoost_model_seq_prediction_2024-08-27_11:02:46_time_hit.csv'},

    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time_hit.csv'},
    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time_hit.csv'},
    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time_hit.csv'},
    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_120s_2025-08-27_16:37:56_time_hit.csv'},


    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Logistic Regression(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time_hit.csv'},
    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Gradient Boosting(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time_hit.csv'},
    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_Random Forest(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time_hit.csv'},
    {'input_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time.csv'
    , 'output_file': '/storage/personal/myhwang/119/experiments_results/Topic_model/tf_idf_sc_bs_3waystorage/results/max_w3_XGBoost(tr_idf)_model_seq_prediction_60s_2025-08-27_16:37:51_time_hit.csv'},


]


# 각 파일에 대해 병렬 처리
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_file, file_info['input_file'], file_info['output_file']) for file_info in files_to_process]

    # 작업이 완료될 때까지 기다림
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing all files"):
        future.result()

print("All files have been processed.")
