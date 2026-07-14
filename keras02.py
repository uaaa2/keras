import tensorflow as tf
print(tf.__version__)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# 데이터 추가 or 횟수 추가로 loss 줄일 수 있음
# y = wx + b를 반복하여 w값을 계속 수정하며 loss 값을 줄임.
#머신러닝

#단층 퍼셉트론 -> 신경망처럼 구성한 다중 퍼셉트론 
# 각 노드 를 연결한 파라미터 한 층을 레이어  
#고양이 -> 생선
#inpuy : 생선을 봄 뇌의 뉴런들을 거쳐 침을 흘린다라는  output 출력  
#AI in 머신러닝 in 딥러닝

#1 데이터 
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#완벽한 대치 데이터 y = ax + b

#2 모델구성
model = Sequential()            #순차적인 모델을 만드는 객체
model.add(Dense(1,input_dim=1)) #홉

#3 컴파일, 훈련
#loss와 optimizer 값 설정
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=5000)# x y 데이터 훈련을 epochs번 하겠다.  
#epochs가 지날수록 loss가 줄어든다


#4 평가, 예측
#모델 객체가 3번에서 훈련한 값들로 
result = model.predict(np.array([7]))
print('7의 예측값 : ', result)

