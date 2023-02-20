from PublicDataReader import TransactionPrice
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

service_key = os.getenv("PUBLIC_DATA_PORTAL_SECRET_KEY")
api = TransactionPrice(service_key)

df = api.get_data(
    property_type="단독다가구",
    trade_type="매매",
    sigungu_code="48170",
    start_year_month="202207",
    end_year_month="202301",
)
