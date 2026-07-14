import tensorflow as tf
print(tf.__version__)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1 데이터
# x = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
# x = x.transpose()
# x = x.T 으로 행열 변경가능
x = np.array([[1,7],[2,8],[3,9],[4,10],[5,11],[6,12]]) #x = (6,2)
print(f"{x.shape}")
#exit() 여기까지만 실행
y = np.array([1,2,3,4,5,6])         
print(f"{y.shape}")


#2 모델구성
model = Sequential()    
model.add(Dense(6, input_shape=(2, )))    
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

result = model.predict(np.array([[7,13]])) 
print(f"7의 예측값 {result}")