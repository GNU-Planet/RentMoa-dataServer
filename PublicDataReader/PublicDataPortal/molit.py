"""
국토교통부 Open API
molit(Ministry of Land, Infrastructure and Transport)
- TransactionPrice 클래스: 부동산 실거래가
"""

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
    print(service_key)

