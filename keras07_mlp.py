import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#멀티레이어퍼셉트론
#행무시 열우선 

#1 데이터

x = np.array([[1,7],[2,8],[3,9],[4,10],[5,11],[6,12]]) #x = (6,2)
print(f"{x.shape}")
#input dim = 2 input shape = (2, )
y = np.array([1,2,3,4,5,6])         
print(f"{y.shape}")


#2 모델구성
model = Sequential()    
#model.add(Dense(6, input_dim=2))
model.add(Dense(6, input_shape=(2, )))    #x가 1,2일때 1이다 / 열 컬럼 피처 특성 속성 Attribute / 열 두개를 input으로 받음
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=6)   

#4 평가예측
loss = model.evaluate(x,y)
print(f"loss = {loss}")

result = model.predict(np.array([[7,13]])) #(1,2) x가 2차원이기때문에 output도 두개나와야 함 
print(f"7의 예측값 {result}")