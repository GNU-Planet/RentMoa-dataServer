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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from xgboost import XGBRegressor
from sklearn.model_selection import learning_curve
from datetime import datetime
from sklearn.tree import export_text
import graphviz

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
        print(data[col])

    # 동 이름을 고유한 번호로 매핑하는 딕셔너리 생성
    dong_mapping = {
        '망경동': 1,
        '주약동': 2,
        '강남동': 3,
        '칠암동': 4,
        '본성동': 5,
        '동성동': 6,
        '남성동': 7,
        '인사동': 8,
        '대안동': 9,
        '평안동': 10,
        '중안동': 11,
        '계동': 12,
        '봉곡동': 13,
        '상봉동': 14,
        '봉래동': 15,
        '수정동': 16,
        '장대동': 17,
        '옥봉동': 18,
        '상대동': 19,
        '하대동': 20,
        '상평동': 21,
        '초전동': 22,
        '장재동': 23,
        '하촌동': 24,
        '신안동': 25,
        '평거동': 26,
        '이현동': 27,
        '유곡동': 28,
        '판문동': 29,
        '귀곡동': 30,
        '가좌동': 31,
        '호탄동': 32,
        '충무공동': 33
    }

    # dong 열의 동 이름을 번호로 매핑하여 변환
    data['dong'] = data['dong'].map(dong_mapping)

    drop_columns = ['contract_start_date', 'id', 'contract_type', 'renewal_request', 'regional_code']
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
    print("훈련 세트 정확도: {:.3f}".format(model.score(X_train, y_train)))
    print("테스트 세트 정확도: {:.3f}".format(model.score(X_test, y_test)))

def visualize_feature_importance(model, feature_names):
    # 모델의 특성 중요도 가져오기
    feature_importance = model.feature_importances_

    # 특성 중요도를 내림차순으로 정렬하여 인덱스 가져오기
    sorted_idx = np.argsort(feature_importance)[::-1]

    # 정렬된 특성 중요도에 따라 특성 이름 정렬
    sorted_feature_names = [feature_names[i] for i in sorted_idx]

    # 특성 중요도 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), sorted_feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance of RandomForestRegressor')
    plt.show()

# 학습 곡선 그리기
def plot_learning_curve(model, X, y, cv, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes)

    # 훈련 세트의 평균 정확도와 표준 편차 계산
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # 검증 세트의 평균 정확도와 표준 편차 계산
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 학습 곡선 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])
    plt.title('Learning Curve')
    plt.show()

def main():
    contract_data = load_data()
    contract_data = preprocess_data(contract_data)

    # 예측할 레이블(y_target)과 특성(X_features) 분리
    y_target = contract_data['contract_end_date']
    X_features = contract_data.drop(['contract_end_date'], axis=1, inplace=False)
    ohe_columns = ['building_id', 'dong']
    X_features_ohe = pd.get_dummies(X_features, columns=ohe_columns)
    print(X_features_ohe.columns)

    #원-핫 인코딩이 적용된 피처 데이터 세트 기반으로 학습/예측 데이터 분할 
    X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target, test_size=0.2, random_state=0)

    dtc_reg = GradientBoostingRegressor(random_state=0)
    get_model_predict(dtc_reg, X_train, X_test, y_train, y_test)    

    # 학습 곡선 그리기
    cv = 5  # 교차 검증 폴드 수
    train_sizes = np.linspace(0.1, 1.0, 10)  # 학습 데이터 크기의 비율
    plot_learning_curve(dtc_reg, X_train, y_train, cv=cv, train_sizes=train_sizes)

    #visualize_feature_importance(dtc_reg, X_features_ohe.columns)
    

if __name__ == "__main__":
    main()