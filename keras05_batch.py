import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np


#1 데이터  
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2 모델구성
model = Sequential()    
model.add(Dense(6, input_dim=1))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일, 훈련
#훈련간 데이터의 양이 많은 경우
#y= wx + b에서 x*w의 양이 많을시 효율적이지 못함
#전체 데이터를 나눠서 대치 계산 2개씩 나눠서 하겠다. 123456훈련 -> 12훈련, 34훈련, 56훈련 1/1 -> 3/3
#기존에 한번에 작업 할 것을 나눈 만큼 훈련을 더하기 때문에 성능이 좋을 수 있음
#tensorflow defult batchsize = 32
#적당한 epochs와 batch를 찾아야함
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)   

#4 평가예측
loss = model.evaluate(x,y)
print(f"loss = {loss}")

result = model.predict(np.array([7]))
print('7의 예측값 : ', result)