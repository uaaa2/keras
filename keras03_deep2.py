import tensorflow as tf
print(tf.__version__)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np


#1 데이터 
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#완벽한 대치 데이터 y = ax + b

#2 모델구성
model = Sequential()            #순차적인 모델을 만드는 객체
#output 차원, input 차원 수 / 하이퍼 파라미터 튜닝

model.add(Dense(6, input_dim=1))#0
model.add(Dense(7))#0
model.add(Dense(4))#0
model.add(Dense(6))#0

model.add(Dense(1))#0

#model에 촘촘한 층 Dense를 add함 

#3 컴파일, 훈련
#loss와 optimizer 값 설정
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)# x y 데이터 훈련을 epochs번 하겠다.  
#epochs가 지날수록 loss가 줄어든다


#4 평가, 예측 loss로 평가
#모델 객체가 3번에서 훈련한 값들로 
result = model.predict(np.array([7]))
print('7의 예측값 : ', result)

