"""
국토교통부 Open API
molit(Ministry of Land, Infrastructure and Transport)
- TransactionPrice 클래스: 부동산 실거래가
"""

import pandas as pd
import requests
import xmltodict
from PublicDataReader.config.database import engine


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
                "매매": {
                    "url": "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev",
                    "columns": [
                        "지역코드",
                        "도로명",
                        "법정동",
                        "지번",
                        "아파트",
                        "건축년도",
                        "층",
                        "전용면적",
                        "년",
                        "월",
                        "일",
                        "거래금액",
                        "도로명건물본번호코드",
                        "도로명건물부번호코드",
                        "도로명시군구코드",
                        "도로명일련번호코드",
                        "도로명지상지하코드",
                        "도로명코드",
                        "법정동본번코드",
                        "법정동부번코드",
                        "법정동시군구코드",
                        "법정동읍면동코드",
                        "법정동지번코드",
                        "일련번호",
                        "거래유형",
                        "중개사소재지",
                        "해제사유발생일",
                        "해제여부",
                    ],
                },
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
                },
            },
            "오피스텔": {
                "매매": {
                    "url": "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcOffiTrade",
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
                        "거래금액",
                        "거래유형",
                        "중개사소재지",
                        "해제사유발생일",
                        "해제여부",
                    ],
                },
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
                },
            },
            "단독다가구": {
                "매매": {
                    "url": "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcSHTrade",
                    "columns": [
                        "지역코드",
                        "법정동",
                        "지번",
                        "주택유형",
                        "건축년도",
                        "대지면적",
                        "연면적",
                        "년",
                        "월",
                        "일",
                        "거래금액",
                        "거래유형",
                        "중개사소재지",
                        "해제사유발생일",
                        "해제여부",
                    ],
                },
                "전월세": {
                    "url": "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcSHRent",
                    "columns": [
                        "지역코드",
                        "법정동",
                        "건축년도",
                        "계약면적",
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
                },
            },
            "연립다세대": {
                "매매": {
                    "url": "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcRHTrade",
                    "columns": [
                        "지역코드",
                        "법정동",
                        "지번",
                        "연립다세대",
                        "건축년도",
                        "층",
                        "대지권면적",
                        "전용면적",
                        "년",
                        "월",
                        "일",
                        "거래금액",
                        "거래유형",
                        "중개사소재지",
                        "해제사유발생일",
                        "해제여부",
                    ],
                },
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
                },
            },
        }
        self.integer_columns = ['년', '월', '일', '층', '건축년도',
                                '거래금액', '보증금액', '보증금', '월세금액', '월세', '종전계약보증금', '종전계약월세']
        self.float_columns = ['전용면적', '대지권면적',
                              '대지면적', '연면적', '계약면적', '건물면적', '거래면적']

    def get_data(self, property_type, trade_type, sigungu_code, year_month=None):
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

        # 단일 연월로 조회
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
                    df[col] = pd.to_numeric(df[col].apply(
                        lambda x: x.strip().replace(",", "") if x is not None and not pd.isnull(x) else x)).astype("Int64")
            for col in self.float_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
        except Exception as e:
            raise Exception(e)

        # 데이터 저장
        conn = engine.connect()
        df.to_sql(name=property_type+trade_type, con=engine, if_exists='append')
        
        
