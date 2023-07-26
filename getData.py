from PublicDataReader import TransactionPrice
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

service_key = os.getenv("PUBLIC_DATA_PORTAL_SECRET_KEY")
api = TransactionPrice(service_key)

df = api.get_data(
    property_type="오피스텔",
    trade_type="전월세",
    sigungu_code="48170",
    start_year_month="202101",
    end_year_month="202306",
)

#api.save_info_data(df, property_type="오피스텔")
api.save_contract_data(df, property_type="오피스텔", trade_type="전월세")
