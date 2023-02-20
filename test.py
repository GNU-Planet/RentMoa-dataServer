from PublicDataReader.config.database import conn
import pymysql

# Connection 으로부터 Cursor 생성
curs = conn.cursor(pymysql.cursors.DictCursor)

# SQL문 실행
sql = "select * from 단독다가구매매"
curs.execute(sql)
 
# 데이타 Fetch
rows = curs.fetchall()
print(rows)     # 전체 rows
