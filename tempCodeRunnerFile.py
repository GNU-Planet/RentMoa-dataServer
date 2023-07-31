    if('contract_end_date' in data.columns):
        print("삭제")
        data.drop('id', axis=1, inplace=True)