import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#output 추가

#전체 데이터훈련이 아닌 70%만 훈련, 나머지는 평가용 
#정확도 1.0이 만점 
#평가용 데이터가 훈련용 데이터보다 높으면 잘못된 가능성이 높다.
#treining data , test data 
#   

#1 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])    
# y = np.array([1,2,3,4,5,6,7,8,9,10])      

#맨 끝에 ,를 찍어도 오류 안남
#훈련용 데이터
x_train = np.array([1,2,3,4,5,6,7,])
y_train = np.array([1,2,3,4,5,6,7,])

#평가용 데이터
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

print(f"{x_train.shape}, {x_test.shape}") # (7,), (3,)  
print(f"{y_train.shape}, {y_test.shape}") # (7,), (3,)
# 가중치 나옴

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


#4 평가예측

# fit()으로 x_train, y_train을 epochs번 훈련해서 각층의 최종 weight , 최종 bias 값 계산
# evaluate(x_test,y_test) 는 최종 weight, bais로 x_train에 대한 예측값을 나오게 함 
# y예측값 = (최종 weight) * x_test + (최종 bias)
# 여기서 계산된 y 예측값을 y_test와 비교하여 loss값이 나오게 됨

loss = model.evaluate(x_test,y_test)   
print(f"loss = {loss}")

result = model.predict(np.array([[11]])) 
print(f"11에 예측값 :{result}")