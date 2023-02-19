# 라이브러리 import
import requests
import json
import xmltodict
import pandas as pd
from pandas.io.json import json_normalize
from tabulate import tabulate

from engine import database

# url 입력
url = ''

# url 불러오기
response = requests.get(url)

contents = response.text

jsonStr = json.dumps(xmltodict.parse(contents), indent=4)
 
dict = json.loads(jsonStr)

df=pd.json_normalize(dict['response']['body']['items']['item'])

df.to_sql(name='RowHouseRent', con=engine, if_exists='append')  