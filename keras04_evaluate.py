import tensorflow as tf
print(tf.__version__)


from keras.models import Sequential
from keras.layers import Dense

import numpy as np


#1 데이터 
#데이터전처리과정
#욜로데이터라벨링
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])


#2 모델구성
model = Sequential()    #
#output 차원, input 차원 수 / 하이퍼 파라미터 튜닝
model.add(Dense(6, input_dim=1))#0
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(1))

#model에 촘촘한 층 Dense를 add함 

#3 컴파일, 훈련
#loss와 optimizer 값 설정 mse 방식으로 계산
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)   
# x y 데이터 훈련을 epochs번 하겠다.  
#epochs가 지날수록 loss가 줄어든다
#순전파 -> 역전파 loss값 출력, 갱신 = 1epoch
#loss값으로 weight갱신, 다음 epoch start

#4 평가, 예측 
# 값이 아닌 loss로 평가 
# 최소의 loss를 지향 최적의 가중치, Weight
loss = model.evaluate(x,y)
#y=wx+b 에서 나온 최종y와 원값y를 비교 최종weight에 대입
print(f"loss = {loss}")
result = model.predict(np.array([7]))
print('7의 예측값 : ', result)

