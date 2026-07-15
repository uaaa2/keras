from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#[넘파이 리스트의 슬라이싱 7:3으로 자르기]
x = np.array([1,2,3,4,5,6,7,8,9,10])    
y = np.array([1,2,3,4,5,6,7,8,9,10])   

#훈련용 데이터
x_train = x[:7]
y_train = y[:7]

#평가용 데이터
x_test = x[7:]
y_test = y[7:]



#2 모델구성
model = Sequential()    
model.add(Dense(100, input_shape=(1, ))) #(None, 1)
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1)) 

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1) 

loss = model.evaluate(x_test,y_test)   
print(f"loss = {loss}")

result = model.predict(np.array([11])) 
print(f"11에 예측값 :{result}")

