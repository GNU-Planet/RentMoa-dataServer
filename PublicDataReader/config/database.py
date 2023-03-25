# MySQL Connector using pymysql
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

sqlalchemy_DB = {
    "drivername": "mysql",
    "host": os.getenv("DB_HOST"),
    "port": 3306,
    "username": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "richJinju",
    "query": {"charset": "utf8"},
}

pymysql_DB = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
    "db": "richJinju",
    "charset": "utf8",
}

engine = create_engine(URL(**sqlalchemy_DB))
conn = pymysql.connect(**pymysql_DB)
