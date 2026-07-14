import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#멀티레이어퍼셉트론

#1 데이터
#이차원리스트  x, y
x = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
#batch= 2일때 1,2 = 1 / 3,4 = 2
y = np.array([1,2,3,4,5,6])


#2 모델구성
model = Sequential()    
model.add(Dense(6, input_dim=2))    #x가 1,2일때 1이다
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)   

#4 평가예측
loss = model.evaluate(x,y)
print(f"loss = {loss}")

result = model.predict(np.array([7,13])) #x가 2차원이기때문에 output도 두개나와야 함 
print('7의 예측값 : ', result)