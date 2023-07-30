import numpy as np
from PublicDataReader.config.database import engine
import pandas as pd
from sqlalchemy import text
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
from datetime import datetime

import warnings 
warnings.filterwarnings("ignore", category = RuntimeWarning)

# 폰트 적용
plt.rcParams['font.family'] = 'AppleGothic'

# 에폭값을 날짜로 변환하는 함수
def convert_epoch_to_date(epoch):
    return pd.to_datetime(epoch, unit='s')

# 데이터를 불러오는 함수
def load_data():
    conn = engine.connect()
    query = """
            SELECT *
            FROM offi_rent_contract
            """

    result = conn.execute(text(query)).fetchall()
    contract_data = pd.DataFrame(result)

    #contract_end_date가 None인 행 제거
    contract_data = contract_data.dropna(subset=['contract_end_date'])
    
    return contract_data

# 데이터 전처리 함수
def preprocess_data(data):
    # 날짜 열을 datetime 객체로 변환
    date_columns = ['contract_date', 'contract_end_date']

    for col in date_columns:
        data[col] = pd.to_datetime(data[col])
        data[col] = data[col].map(pd.Timestamp.timestamp)

    # contract_type 열을 전처리 (0=신규, 1=갱신, 2=null)
    data['contract_type'] = data['contract_type'].apply(lambda x : 0 if x == '신규' else 1 if x == '갱신' else 2)

    # renewal_request 열을 전처리 (0=null, 1=사용)
    data['renewal_request'] = data['renewal_request'].apply(lambda x : 0 if x == 'null' else 1)

    drop_columns = ['dong', 'contract_start_date', 'id']
    data.drop(drop_columns, axis=1, inplace=True)

    # (contract_end_date - contract_date)로 duration 계산
    data['duration'] = data['contract_end_date'] - data['contract_date']
    # 계약 기간을 월(Month) 로 변환
    data['duration_months'] = (round(data['duration'] / (60 * 60 * 24 * 30))).astype(int)
    # 계약 기간이 11, 12, 13, 23, 24, 25개월인 데이터만 추출
    data = data[data['duration_months'].isin([11, 12, 13, 23, 24, 25])]
    data.drop(['duration', 'duration_months'], axis=1, inplace=True)

    return data

# log 값 변환 시 NaN 등의 이슈로 log()가 아닌 log1p()를 이용해 RMSLE 계산 
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

#MSE, RMSE, RMSLE를 모두 계산
def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    #MAE는 사이킷런의 mean_absolute_error()로 계산
    mae_val = mean_absolute_error(y, pred)
    print("RMSLE: {0:.3f}, RMSE : {1:.3f}, MAE : {2:.3f}".format(rmsle_val, rmse_val, mae_val))

# 실제값과 예측 값이 어느 정도 차이가 나는지 DF의 칼럼으로 만들어서 오류 값이 가장 큰 순으로 5개 확인 
def get_top_error_data(y_test, pred, n_tops = 5):
    #DF의 칼럼으로 실제 대여 횟수(Count)와 예측값을 서로 비교할 수 있도록 생성. 
    result_df = pd.DataFrame(y_test.values, columns = ['real_count'])
    result_df['predicted_count'] = np.round(pred)
    result_df['diff'] = ((result_df['real_count'] - result_df['predicted_count']) / (60 * 60 * 24 * 30)).astype(int)
    result_df['real_count'] = convert_epoch_to_date(result_df['real_count'])
    result_df['predicted_count'] = convert_epoch_to_date(result_df['predicted_count'])

    #예측값과 실제 값이 가장 큰 데이터 순으로 출력 
    print(result_df.sort_values('diff', ascending = False)[:n_tops])

    #1개월 이내의 차이 값 % 출력
    print(sum(result_df['diff'] == 0) / len(result_df))
    print(sum(result_df['diff'] <= 1) / len(result_df))

    # 차이값 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(result_df)), result_df['diff'])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Difference (Months)')
    plt.title('Difference between Predicted and Actual End Date')
    plt.show()

#모델과 학습/ 테스트 데이터 세트를 입력하면 성능 평가 수치를 반환 
def get_model_predict(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("###", model.__class__.__name__, '###')
    evaluate_regr(y_test, pred)
    get_top_error_data(y_test, pred, n_tops=25)

def main():
    contract_data = load_data()
    contract_data = preprocess_data(contract_data)

    # 예측할 레이블(y_target)과 특성(X_features) 분리
    y_target = contract_data['contract_end_date']
    X_features = contract_data.drop(['contract_end_date'], axis=1, inplace=False)
    print(X_features.columns)
    ohe_columns = ['contract_type', 'renewal_request', 'building_id', 'regional_code']
    X_features_ohe = pd.get_dummies(X_features, columns=ohe_columns)

    #원-핫 인코딩이 적용된 피처 데이터 세트 기반으로 학습/예측 데이터 분할 
    X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target, test_size=0.2, random_state=0)

    dtc_reg = DecisionTreeClassifier(random_state=0)
    get_model_predict(dtc_reg, X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    main()