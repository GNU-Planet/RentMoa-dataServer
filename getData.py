from PublicDataReader import TransactionPrice
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

service_key = os.getenv("PUBLIC_DATA_PORTAL_SECRET_KEY")
api = TransactionPrice(service_key)

df = api.get_data(
    property_type="단독다가구",
    trade_type="전월세",
    sigungu_code="48170",
    year_month="202101",
)

#api.save_info_data(df, property_type="연립다세대")
api.save_contract_data(df, property_type="단독다가구")
