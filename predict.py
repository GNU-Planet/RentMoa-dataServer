import pandas as pd
from PublicDataReader.config.database import engine
from sqlalchemy import text

conn = engine.connect()
query = text(f"SELECT * FROM offi_rent_contract")

result = conn.execute(query).fetchall();
contract_data = pd.DataFrame(result)

# contract_end_date가 None인 행 제거
contract_data = contract_data.dropna(subset=['contract_end_date'])

columns = ['contract_area', 'floor']

# 날짜 열을 datetime 객체로 변환
date_columns = ['contract_date', 'contract_start_date', 'contract_end_date']
for col in date_columns:
    contract_data[col] = pd.to_datetime(contract_data[col])
    
    contract_data[col] = contract_data[col].map(pd.Timestamp.timestamp)
contract_data['contract_duration_days'] = (contract_data['contract_end_date'] - contract_data['contract_start_date'])
contract_data['contract_start_to_date_days'] = (contract_data['contract_start_date'] - contract_data['contract_date'])

columns += ['contract_date', 'contract_duration_days', 'contract_start_to_date_days']

# 예측해야할 변수(계약종료일)
y = contract_data['contract_end_date']
# 예측하는 데에 사용되는 변수열들(법정동, 건물, 갱신 여부, 계약년도, year, month, 면적, 보증금, 월세, 층 등)
contract_features = columns
X = contract_data[contract_features]


from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터를 학습 데이터와 평가 데이터로 나누기 (test_size=0.2는 평가 데이터의 비율을 20%로 지정)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 모델을 정의합니다. 매번 똑같은 결과를 얻기 위해 random_state에 숫자를 지정합니다.
contract_model = DecisionTreeRegressor(random_state=1)

# 학습 데이터로 모델을 학습시킵니다.
contract_model.fit(X_train, y_train)

# 변수 중요도를 얻습니다.
feature_importances = contract_model.feature_importances_

# 변수 중요도를 출력합니다.
print("Variable Importance:")
for i, importance in enumerate(feature_importances):
    print(f"Feature {X.columns[i]}: {importance}")

# 평가 데이터로 예측을 수행합니다.
predictions = contract_model.predict(X_test)
predicted_dates = [datetime.fromtimestamp(epoch_value).strftime('%Y-%m') for epoch_value in predictions]
actual_dates = [datetime.fromtimestamp(epoch_value).strftime('%Y-%m') for epoch_value in y_test]


print("모델의 예측 값:")
print(predicted_dates)

# 실제 y값 출력 (평가 데이터의 실제 contract_end_date 값)
print("실제 값:")
print(actual_dates)

# 정확도를 계산하여 출력합니다.
accuracy = accuracy_score(actual_dates, predicted_dates)
print(f"모델의 정확도: {accuracy}")

# 정확도를 계산하여 출력합니다.
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

