import pandas as pd
from PublicDataReader.config.database import engine
from sqlalchemy import text
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

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
    # contract_end_date가 None인 행 제거
    contract_data = contract_data.dropna(subset=['contract_end_date'])
    return contract_data

# 데이터 전처리 함수
def preprocess_data(data):
    # 날짜 열을 datetime 객체로 변환
    date_columns = ['contract_date', 'contract_end_date']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])
        data[col] = data[col].map(pd.Timestamp.timestamp)

    return data

# 모델을 학습시키는 함수
def train_model(X, y):
    # 데이터를 학습 데이터와 평가 데이터로 나누기 (test_size=0.2는 평가 데이터의 비율을 20%로 지정)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # 모델을 정의합니다. 매번 똑같은 결과를 얻기 위해 random_state에 숫자를 지정합니다.
    contract_model = DecisionTreeRegressor(random_state=1)
    # 학습 데이터로 모델을 학습시킵니다.
    contract_model.fit(X_train, y_train)
    return contract_model, X_test, y_test

# 모델을 사용하여 계약종료일을 예측하는 함수
def predict_dates(model, X_test):
    predictions = model.predict(X_test)
    predicted_dates = [datetime.fromtimestamp(epoch_value).strftime('%Y-%m') for epoch_value in predictions]
    return predicted_dates

# 모델의 정확도를 평가하는 함수
def evaluate_model(actual_dates, predicted_dates):
    accuracy = accuracy_score(actual_dates, predicted_dates)
    print(f"모델의 정확도: {accuracy}")
    correct_count = 0
    tolerance = timedelta(days=30)  # 1개월 오차범위 설정
    date_format = "%Y-%m";
    for actual, predicted in zip(actual_dates, predicted_dates):
        actual = datetime.strptime(actual, date_format)
        predicted = datetime.strptime(predicted, date_format)
        if abs(actual - predicted) <= tolerance:
            correct_count += 1
    accuracy = correct_count / len(actual_dates)
    print(f"모델의 정확도(1개월 오차범위 허용): {accuracy}")

# 변수 중요도를 출력하는 함수
def print_feature_importances(model, X):
    feature_importances = model.feature_importances_
    features = X.columns
    for feature, importance in zip(features, feature_importances):
        print(f"{feature}: {importance}")

def main():
    test_type = ["무관"]
    for type in test_type:
        contract_data = load_data(type)
        contract_data = preprocess_data(contract_data)
        
        # 모델에 사용할 feature와 target 변수 설정
        contract_features = ['deposit', 'monthly_rent', 'contract_date']
        X = contract_data[contract_features]
        y = contract_data['contract_end_date']

        # # 교차 검증 수행 (cv=5는 5-fold 교차 검증을 의미)
        # model = DecisionTreeRegressor(random_state=1)
        # scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        # # 교차 검증 결과 출력
        # print(f"----- {type} ------")
        # print(f"사용된 features: {contract_features}")
        # print('교차 검증별 R-squared : {}'.format(np.round(scores, 4)))

        # 모델 학습
        #model.fit(X, y)

        # 변수 중요도 출력
        # print("변수 중요도:")
        # print_feature_importances(model, X)
        
        model, X_test, y_test = train_model(X, y)
        predicted_dates = predict_dates(model, X_test)
        actual_dates = [datetime.fromtimestamp(epoch_value).strftime('%Y-%m') for epoch_value in y_test]

        print(f"----- {type} ------")
        print(f"사용된 features: {contract_features}")
        evaluate_model(actual_dates, predicted_dates)
        print_feature_importances(model, X)
        

if __name__ == "__main__":
    main()
