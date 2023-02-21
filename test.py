from PublicDataReader.config.database import conn
import pymysql
import pandas as pd

# Connection 으로부터 Cursor 생성
curs = conn.cursor(pymysql.cursors.DictCursor)

# SQL문 실행
sql = "select * from 단독다가구전월세 where 월세금액!=0"
curs.execute(sql)
 
# 데이타 Fetch
rows = curs.fetchall()
delivery = pd.DataFrame(rows)

print(delivery.groupby("법정동").size())
#print(delivery.groupby(['월', '법정동'])[])
