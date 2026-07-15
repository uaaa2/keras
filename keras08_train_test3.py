from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


#[슬라이싱 ] train 과 test를 섞어서 랜덤하게 7:3으로 뽑는다.
#[넘파이 리스트의 슬라이싱 7:3으로 자르기]
#슬라이싱간 범위가 큰 데이터를 슬라이싱을 잘못하면 나머지 부분이 반영이 안될 수 있음
#랜덤으로 퍼센트를 뽑는다. 
x = np.array([1,2,3,4,5,6,7,8,9,10])    
y = np.array([1,2,3,4,5,6,7,8,9,10])   
    
#훈련용 데이터


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                   train_size=0.7,
                                   shuffle=True,   #True면 섞음, False면 리스트 순서대로 
                                   random_state = 1,  #n번 난수표를 기준으로 ramdom하게 돌림
                                   )


print(x_train , x_test)
print(y_train , y_test)
exit()
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
model.fit(x_train,y_train,epochs=100, ) 

loss = model.evaluate(x_test,y_test)   
print(f"loss = {loss}")

result = model.predict(np.array([11])) 
print(f"11에 예측값 :{result}")

