import requests
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

def address_to_coordinate(dong, jibun):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {
        "analyze_type": "similar",
        "page": 1,
        "size": 10,
        "query": "경남 진주시 " + dong + " " + jibun
    }
    headers = {
        "Authorization": "KakaoAK " + os.getenv("KAKAO_REST_API_KEY")
    }

    response = requests.get(url, params=params, headers=headers)
    building_lat = response.json()['documents'][0]['address']['y'] 
    building_lng = response.json()['documents'][0]['address']['x']

    return building_lat, building_lng