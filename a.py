

#mongodb연결
import pymongo
from pymongo.server_api import ServerApi
from pymongo.mongo_client import MongoClient
client = pymongo.MongoClient("mongodb+srv://sladurrkq:1234@cluster0.bgrkr.mongodb.net/project3_?retryWrites=true&w=majority", server_api=ServerApi('1'))
db = client.test


#db만들기
DATABASE = 'myFirstDatabase'
database = client[DATABASE]

COLLECTION = 'whale-collection'
collection = database[COLLECTION]


#data 가져오기 로컬
data_path = '/Users/jj/Desktop/pj3/Data/features_3_sec.csv'

import pandas as pd
df = pd.read_csv(data_path)

#필요없는열 제거
df = df.drop(columns=['filename', 'length'])


#X,y로 나눔
X = df.drop(columns=['label'])
y = df['label']


#list기준으로 df를 dict으로 변환 
df = df.to_dict('list')
#mongodb에 업로드
collection.insert_one(df)


#전처리
import sklearn
from sklearn.preprocessing import MinMaxScaler


scaler = sklearn.preprocessing.MinMaxScaler()
np_scaled = scaler.fit_transform(X)

X = pd.DataFrame(np_scaled, columns=X.columns)


#데이터셋분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#모델학습
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=20)
model.fit(X_train,y_train)


#예측
predition = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predition)) #0.66~~~~~~


#모델저장
import joblib
joblib.dump(model, './model.pkl')


#X_test저장
X_test.to_csv('X_test.csv')


print(db.wow.find({}))