import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#x 행열 변경
#1 데이터

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,4,3,2,1,0]
              ])      #(3,10) 3열 10행

x = x.T #(10,3) y의 행과 x의 열을 맞춰야하기 때문에 transpose()
print(f"{x.shape}") 

y = np.array([1,2,3,4,5,6,7,8,9,10])      #(10,)    
print(f"{y.shape}")


#2 모델구성
model = Sequential()    
model.add(Dense(100, input_shape=(3, )))    # (None,3)
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)   

#4 평가예측
loss = model.evaluate(x,y)
print(f"loss = {loss}")

result = model.predict(np.array([[10,1.3,0]])) # (1,3)
print(f"[[10,1.3,0]]예측값 :{result}")