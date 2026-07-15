import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#output 추가

#전체 데이터훈련이 아닌 70%만 훈련, 나머지는 평가용 
#Statement seems to have no effect : 정확도 1.0이 만점 
#평가용 데이터가 훈련용 데이터보다 높으면 잘못된 가능성이 높다.
#treining data , test data 
#   

#1 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]
              ])    #(3,10)
x = x.T             #(10,3)
print(f"{x.shape}") 
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]])      #(2,10)
y = y.T    #(10,2)
print(f"{y.shape}") 

#2 모델구성
model = Sequential()    
model.add(Dense(100, input_shape=(3, )))    # (None,3)
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(2)) #output (N, 2) (None,2) 형태

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)   

#4 평가예측
loss = model.evaluate(x,y)   
print(f"loss = {loss}")

result = model.predict(np.array([[10,1.3,0]]))  #(10,2) (10,3)에 대한 계산  
print(f"[[10,1.3,0]]예측값 :{result}")