import requests
import os
from dotenv import load_dotenv
from config.database import engine
from sqlalchemy import text

load_dotenv(verbose=True)


class Add2Loc:
    def __init__(self):
        self.meta_dict = {
            "table_name": {
                "아파트": "apartment_info",
                "오피스텔": "offi_info",
                "연립다세대": "row_house_info"
            },

        }
        self.conn = engine.connect()

    # 주소를 입력받아 위도, 경도를 반환하는 함수

    def address_to_coordinate(self, dong, jibun):
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
        try:
            response = requests.get(url, params=params, headers=headers)
            building_lat = response.json()['documents'][0]['address']['y']
            building_lng = response.json()['documents'][0]['address']['x']
            print(f"{dong} {jibun}의 위도:{building_lat}, 경도:{building_lng}")

        except:
            building_lat, building_lng = 'wrong_address', 'wrong_address'
        finally:
            return building_lat, building_lng

    # 건물 정보에 위도, 경도를 추가하는 함수
    def update_buildings(self, property_type):
        if property_type in self.meta_dict["table_name"]:
            table_name = self.meta_dict["table_name"][property_type]
            # property_type 테이블의 위도경도가 없는 데이터를 불러옵니다.
            try:
                query = text(
                    f"SELECT * FROM {table_name} WHERE building_lat IS NULL OR building_lng IS NULL;")
                result = self.conn.execute(query)
                print(f"{table_name} 조회")
                buildings = result.fetchall()
                print(buildings)
            except Exception as e:
                print(f"{table_name} 조회에 실패하였습니다. 에러메시지: {e}")

            updated_buildings = []

            # 위도 및 경도를 갱신한 데이터를 생성합니다.
            for building in buildings:
                dong, jibun = building[2], building[3]
                building_lat, building_lng = self.address_to_coordinate(
                    dong, jibun)
                updated_building = (*building[:6], building_lat, building_lng)
                updated_buildings.append(updated_building)

            return updated_buildings
        else:
            return "Wrong property type"

    # 좌표가 추가된 건물정보를 DB에 저장하는 함수
    def save_updated_buildings(self, updated_buildings, property_type):
        # 수정할 건물 정보

        if not updated_buildings:
            print("수정할 건물 정보가 없습니다.")
            return
        if property_type in self.meta_dict["table_name"]:
            table_name = self.meta_dict["table_name"][property_type]

            # 갱신된 데이터를 DB에 저장합니다.
            for updated_building in updated_buildings:
                id = updated_building[0]
                building_lat = updated_building[6]
                building_lng = updated_building[7]

                try:
                    query = text(
                        f"UPDATE {table_name} SET building_lat = :building_lat, building_lng = :building_lng WHERE id = :id;")
                    self.conn.execute(
                        query, {"building_lat": building_lat, "building_lng": building_lng, "id": id})
                    print(f"{table_name}의 {id}번 데이터가 갱신되었습니다.")

                except Exception as e:
                    print(f"{table_name}의 데이터 갱신에 실패하였습니다. 에러메시지: {e}")

            self.conn.commit()
        else:
            return "Wrong property type"


# 아파트
add2loc = Add2Loc()
updated_buildings = add2loc.save_updated_buildings(
    add2loc.update_buildings("아파트"), "아파트")
