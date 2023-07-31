from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from PublicDataReader.config.database import engine
from sqlalchemy import text
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 에폭값을 날짜로 변환하는 함수
def convert_epoch_to_date(epoch):
    return pd.to_datetime(epoch, unit='s')

# 데이터를 불러오는 함수
def load_data(type="무관|전세|월세"):
    conn = engine.connect()
    query = "SELECT * FROM offi_rent_contract"

    if type == "전세":
        query += " WHERE monthly_rent = 0"
    elif type == "월세":
        query += " WHERE monthly_rent != 0"

    result = conn.execute(text(query)).fetchall()
    contract_data = pd.DataFrame(result)
    
    # contract_end_date가 None인 행과 not null인 행 분리
    contract_data_null = contract_data[contract_data['contract_end_date'].isnull()]
    contract_data_not_null = contract_data.dropna(subset=['contract_end_date'])
    
    return contract_data_null, contract_data_not_null

# 데이터 전처리 함수
def preprocess_data(data):
    # 날짜 열을 datetime 객체로 변환
    date_columns = ['contract_date', 'contract_start_date', 'contract_end_date']
    # contract_date의 일자를 모두 1로 바꾸기
    data['contract_date'] = data['contract_date'].apply(lambda x: x.replace(day=1))
    try:
        for col in date_columns:
            data[col] = pd.to_datetime(data[col])
            data[col] = data[col].map(pd.Timestamp.timestamp)
    except:
        pass

    # (contract_end_date - contract_date)로 duration 계산
    data['duration'] = data['contract_end_date'] - data['contract_date']

    # 계약 기간을 월(Month) 로 변환
    data['duration_months'] = (round(data['duration'] / (60 * 60 * 24 * 30))).astype(int)

    # 계약 기간이 11, 12, 13, 23, 24, 25개월인 데이터만 추출
    data = data[data['duration_months'].isin([11, 12, 13, 23, 24, 25])]

    return data

# 모델을 학습시키는 함수 (랜덤 포레스트)
def train_model_random_forest(X, y):
    # 데이터를 학습 데이터와 평가 데이터로 나누기 (test_size=0.2는 평가 데이터의 비율을 20%로 지정)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤포레스트 모델 생성 및 학습
    rf_model = RandomForestRegressor(random_state=42)
    
    rf_model.fit(X_train, y_train)
    return rf_model, X_test, y_test

# 모델을 학습시키는 함수 (DecisionTreeClassifier)
def train_model_decision_tree(X, y):
    # 데이터를 학습 데이터와 평가 데이터로 나누기 (test_size=0.2는 평가 데이터의 비율을 20%로 지정)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 의사결정 트리 모델 생성
    clf = DecisionTreeClassifier(random_state=42)

    clf.fit(X_train, y_train)

    return clf, X_test, y_test

# DB에 데이터를 업데이트하는 함수
def update_contract_dates(data):
    # Create a connection to the database
    conn = engine.connect()

    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        # Get the id and contract_date from the row
        idx = row['id']
        contract_end_date = row['contract_end_date']

        # Update the contract_date in the database using an SQL UPDATE query
        query = f"UPDATE offi_rent_contract SET contract_end_date = '{contract_end_date}' WHERE id = {idx}"
        conn.execute(text(query))

        #commit
        conn.commit()

    # Close the database connection
    conn.close()

def main():
    test_type = ["무관"]
    for type in test_type:
        empty_data, contract_data = load_data(type)
        contract_data = preprocess_data(contract_data)
        empty_data = preprocess_data(empty_data)
        
        # 모델에 사용할 feature와 target 변수 설정
        contract_features = ['deposit', 'monthly_rent', 'contract_date', 'contract_area', 'floor']
        X = contract_data[contract_features]
        y = contract_data['contract_end_date']

        # 모델 학습
        model, X_test, y_test = train_model_decision_tree(X, y)

        # 검증 데이터로 모델 평가
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # MSE 계산
        mse = mean_squared_error(y_test, y_pred)

        # 예측값과 실제값을 에폭에서 날짜로 변환
        y_pred_dates = convert_epoch_to_date(y_pred)
        y_test_dates = convert_epoch_to_date(y_test)

        # 데이터프레임으로 변환
        prediction_df = pd.DataFrame({'Actual End Date': y_test_dates, 'Predicted End Date': y_pred_dates})

        # 차이 계산 및 평균값 출력
        prediction_df['Difference (Months)'] = (prediction_df['Predicted End Date'] - prediction_df['Actual End Date']) / pd.Timedelta(days=30)
        # 10개월 이상 차이나는 값 보기
        outlier_data = prediction_df[abs(prediction_df['Difference (Months)']) > 1.1]
        actual_outlier_data = contract_data.loc[outlier_data.index]
        print(actual_outlier_data)

        # 월세와 보증금에 따른 계약 기간의 시각화
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='monthly_rent', y='deposit', hue='duration_months', data=actual_outlier_data, palette='viridis')
        plt.xlabel('월세')
        plt.ylabel('보증금')
        plt.title('월세와 보증금에 따른 계약 기간')
        plt.show()

        # MSE와 평균 차이 출력
        #print(prediction_df)
        print("Mean Squared Error (MSE): {:.2f}".format(mse))

        num_correct = sum(abs(prediction_df['Difference (Months)']) == 0)/len(prediction_df)
        print("정답 퍼센트:", num_correct)
        num_correct = sum(abs(prediction_df['Difference (Months)']) <= 1.1)/len(prediction_df)
        print("1개월 오차 범위 내 정답 퍼센트:", num_correct)
        

if __name__ == "__main__":
    main()
