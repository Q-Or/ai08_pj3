# 플라스크 rest-api 생성
from flask import Flask, request
## 서버 띄우고 접속 허용
app = Flask(__name__)

@app.route("/" , methods=["GET"])
def model():
    #입력 csv 불러오기
    import pandas as pd
    X_test = pd.read_csv('./X_test.csv')

    #pkl로 저장한 모델 불러오기
    import joblib
    loaded_model = joblib.load('./model.pkl')

    #모델입력에 맞게 처리
    X_test = X_test.set_index(X_test['Unnamed: 0'])
    X_test = X_test.drop(columns=['Unnamed: 0'])
    
    # 결과값을 dict으로 바꿈 
    predition = loaded_model.predict(X_test) 
    predition = dict(enumerate(predition))   
    
    # dict predition return
    return predition

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=50, debug=True)  # debug=True causes Restarting with stat