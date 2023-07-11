"""
국토교통부 Open API
molit(Ministry of Land, Infrastructure and Transport)
- TransactionPrice 클래스: 부동산 실거래가
"""

import pandas as pd
import requests
import xmltodict
import datetime
from PublicDataReader.config.database import engine
from tabulate import tabulate
from sqlalchemy import text


class TransactionPrice:
    """
    국토교통부 부동산 실거래가 조회 클래스
    parameters
    ----------
    service_key : str
        국토교통부 Open API 서비스키
    """

    def __init__(self, service_key=None):
        self.service_key = service_key
        self.meta_dict = {
            "아파트": {
                "전월세": {
                    "url": "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent",
                    "columns": [
                        "지역코드",
                        "법정동",
                        "지번",
                        "아파트",
                        "건축년도",
                        "층",
                        "전용면적",
                        "년",
                        "월",
                        "일",
                        "보증금액",
                        "월세금액",
                        "계약구분",
                        "계약기간",
                        "갱신요구권사용",
                        "종전계약보증금",
                        "종전계약월세",
                    ],
                    "table_name": "apartment",
                },
            },
            "오피스텔": {
                "전월세": {
                    "url": "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcOffiRent",
                    "columns": [
                        "지역코드",
                        "시군구",
                        "법정동",
                        "지번",
                        "단지",
                        "건축년도",
                        "층",
                        "전용면적",
                        "년",
                        "월",
                        "일",
                        "보증금",
                        "월세",
                        "계약구분",
                        "계약기간",
                        "갱신요구권사용",
                        "종전계약보증금",
                        "종전계약월세",
                    ],
                    "table_name": "offi",
                },
            },
            "단독다가구": {
                "전월세": {
                    "url": "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcSHRent",
                    "columns": [
                        '지역코드', 
                        '법정동', 
                        '건축년도', 
                        '계약면적', 
                        '년', 
                        '월', 
                        '일', 
                        '보증금액', 
                        '월세금액', 
                        '계약구분', 
                        '계약기간', 
                        '갱신요구권사용', 
                        '종전계약보증금', 
                        '종전계약월세'],
                    "table_name": "detached_house",
                },
            },
            "연립다세대": {
                "전월세": {
                    "url": "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcRHRent",
                    "columns": [
                        "지역코드",
                        "법정동",
                        "지번",
                        "연립다세대",
                        "건축년도",
                        "층",
                        "전용면적",
                        "년",
                        "월",
                        "일",
                        "보증금액",
                        "월세금액",
                        "계약구분",
                        "계약기간",
                        "갱신요구권사용",
                        "종전계약보증금",
                        "종전계약월세",
                    ],
                    "table_name": "row_house",
                },
            },
        }
        self.integer_columns = [
            "년",
            "월",
            "일",
            "층",
            "건축년도",
            "거래금액",
            "보증금액",
            "보증금",
            "월세금액",
            "월세",
            "종전계약보증금",
            "종전계약월세",
        ]
        self.float_columns = ["전용면적", "대지권면적", "대지면적", "연면적", "계약면적", "건물면적", "거래면적"]

    def get_data(
        self,
        property_type,
        trade_type,
        sigungu_code,
        year_month=None,
        start_year_month=None,
        end_year_month=None,
    ):
        """
        부동산 실거래가 조회
        Parameters
        ----------
        property_type : str
            부동산 이름 (ex. 아파트, 오피스텔, 단독다가구, 연립다세대, 토지, 분양입주권, 공장창고등)
        trade_type : str
            거래 유형 (ex. 매매, 전월세)
        sigungu_code : str
            시군구코드 (ex. 11110)
        year_month : str, optional
            조회할 연월 (ex. 201901), by default None
        start_year_month : str, optional
            조회할 시작 연월 (ex. 201901), by default None
        end_year_month : str, optional
            조회할 종료 연월 (ex. 201901), by default None
        verbose : bool, optional
            진행 상황 출력 여부, by default False
        **kwargs : dict
            API 요청에 필요한 추가 인자
        """

        try:
            # 부동산 이름과 거래 유형으로 API URL 선택 (ex. 아파트, 매매)
            url = self.meta_dict.get(property_type).get(trade_type).get("url")
            columns = self.meta_dict.get(property_type).get(trade_type).get("columns")
        except AttributeError:
            raise AttributeError("부동산 이름과 거래 유형을 확인해주세요.")

        # 서비스키, 행수, 시군구코드 설정
        params = {
            "serviceKey": requests.utils.unquote(self.service_key),
            "LAWD_CD": sigungu_code,
        }

        # 기간으로 조회
        if start_year_month and end_year_month:
            start_date = datetime.datetime.strptime(str(start_year_month), "%Y%m")
            start_date = datetime.datetime.strftime(start_date, "%Y-%m")
            end_date = datetime.datetime.strptime(str(end_year_month), "%Y%m")
            end_date += datetime.timedelta(days=31)
            end_date = datetime.datetime.strftime(end_date, "%Y-%m")
            ts = pd.date_range(start=start_date, end=end_date, freq="m")
            date_list = list(ts.strftime("%Y%m"))

            df = pd.DataFrame(columns=columns)
            for year_month in date_list:
                params["DEAL_YMD"] = year_month
                res = requests.get(url, params=params, verify=False)
                res_json = xmltodict.parse(res.text)
                if res_json["response"]["header"]["resultCode"] != "00":
                    error_message = res_json["response"]["header"]["resultMsg"]
                    raise Exception(error_message)
                items = res_json["response"]["body"]["items"]
                if not items:
                    continue
                data = items["item"]
                sub = pd.DataFrame(data)
                df = pd.concat([df, sub], axis=0, ignore_index=True)

        # 단일 연월로 조회
        else:
            df = pd.DataFrame(columns=columns)
            params["DEAL_YMD"] = year_month
            res = requests.get(url, params=params)
            res_json = xmltodict.parse(res.text)
            # 에러 핸들링
            if res_json["response"]["header"]["resultCode"] != "00":
                error_message = res_json["response"]["header"]["resultMsg"]
                raise Exception(error_message)
            items = res_json["response"]["body"]["items"]
            if not items:
                return pd.DataFrame(columns=columns)

            data = items["item"]
            sub = pd.DataFrame(data)
            df = pd.concat([df, sub], axis=0, ignore_index=True)

        # 컬럼 타입 변환
        try:
            for col in self.integer_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].apply(
                            lambda x: x.strip().replace(",", "")
                            if x is not None and not pd.isnull(x)
                            else x
                        )
                    ).astype("Int64")
            for col in self.float_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
        except Exception as e:
            raise Exception(e)

        # 공통 예외처리
        df['계약일'] = pd.to_datetime(df[['년', '월', '일']].astype(str).agg('-'.join, axis=1)).dt.date # 년, 월, 일 -> 계약일
        df = df[~df['법정동'].str.endswith('리')] # 법정동 끝에 '리'가 붙은 행 삭제

        # 계약기간 처리
        for index, row in df.iterrows():
            계약기간 = row['계약기간']
            if pd.isna(계약기간):
                # 계약기간이 None인 경우
                df.at[index, '계약시작일'] = row['계약일']  # 계약시작일 = 계약일
                df.at[index, '계약종료일'] = (row['계약일'] + pd.DateOffset(years=2)).date()  # 2년을 더한 계약종료일
            else:
                # 계약기간이 값이 있는 경우
                계약시작일, 계약종료일 = 계약기간.split('~')
                df.at[index, '계약시작일'] = pd.to_datetime(계약시작일, format='%y.%m').date()
                df.at[index, '계약종료일'] = pd.to_datetime(계약종료일, format='%y.%m').date()
        df.drop(['계약기간', '년', '월', '일', '갱신요구권사용', '종전계약보증금', '종전계약월세'], axis=1, inplace=True)

        # 아파트 예외처리
        if property_type == "아파트":
            df.rename(columns={'아파트': 'building_name', '보증금액': '보증금', '월세금액': '월세'}, inplace=True)

        # 오피스텔 예외처리
        if property_type == "오피스텔":
            df.rename(columns={'단지': 'building_name'}, inplace=True)
            df.drop(['시군구'], axis=1, inplace=True)

        # 연립다세대 예외처리
        if property_type == "연립다세대":
            df.rename(columns={'연립다세대': 'building_name', '보증금액': '보증금', '월세금액': '월세'}, inplace=True)

        # 단독다가구 예외처리
        if property_type == "단독다가구":
            df.rename(columns={'계약면적': '면적', '보증금액': '보증금', '월세금액': '월세'}, inplace=True)

        # 아파트, 오피스텔, 연립다세대 예외처리
        if property_type in ["아파트", "오피스텔", "연립다세대"]:
            df.rename(columns={'전용면적': '면적', '지번': 'jibun', '층': 'floor'}, inplace=True)

        df.rename(columns={
            '지역코드': 'regional_code', 
            '법정동': 'dong', 
            '건축년도': 'build_year', 
            '면적': 'contract_area',
            '보증금': 'deposit', 
            '월세': 'monthly_rent', 
            '계약구분': 'contract_type',
            '계약일': 'contract_date', 
            '계약시작일': 'contract_start_date', 
            '계약종료일': 'contract_end_date'}, inplace=True)

        return df

    # 주택 정보 저장
    def save_info_data(self, df, property_type):
        table_name = self.meta_dict.get(property_type).get("전월세").get("table_name") + "_info"
        selected_df = df[['regional_code', 'dong', 'jibun', 'building_name', 'build_year']].drop_duplicates()
        conn = engine.connect()
        selected_df.to_sql(name=table_name, con=engine, if_exists="append", index=False)
    
    # 계약 데이터 저장
    def save_contract_data(self, df, property_type):
        table_name = self.meta_dict.get(property_type).get("전월세").get("table_name") + "_contract"

        selected_columns = ['regional_code', 'dong', 'contract_type', 'contract_date', 'contract_start_date', 'contract_end_date', 'contract_area', 'deposit', 'monthly_rent']
        
        if property_type in ["아파트", "오피스텔", "연립다세대"]:
            conn = engine.connect()
            query = text("SELECT jibun, id FROM offi_info")
            result = conn.execute(query)
            building_name_to_id = dict(result.fetchall())
            conn.close()

            selected_columns.insert(2, 'building_id')
            selected_columns.insert(7, 'floor')
            df['building_id'] = df['jibun'].map(building_name_to_id)
        else:
            selected_columns.insert(3, 'build_year')

        selected_df = df[selected_columns]
        
        selected_df.to_sql(name=table_name, con=engine, if_exists="append", index=False)