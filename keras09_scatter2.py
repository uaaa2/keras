from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])    
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])   

#훈련용 데이터


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                   train_size=0.7,  
                                   test_size=0.3,  
                                   shuffle=True,   
                                   random_state = 1,  
                                   )                  

print(x_train , x_test)
print(y_train , y_test)

#2 모델구성
model = Sequential()    
model.add(Dense(120, input_shape=(1, ))) #(None, 1)
model.add(Dense(90))
model.add(Dense(64))
model.add(Dense(36))
model.add(Dense(12)) 
model.add(Dense(1)) 

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100) 

loss = model.evaluate(x_test,y_test)    #loss = 7.354154109954834
print(f"loss = {loss}")     

result = model.predict([x]) 
print(f"x에 예측값 :{result}")

#################  그래프 그리기 ########################################
import matplotlib.pyplot as plt
plt.scatter(x,y)    #x와 y를 점 찍음
plt.plot(x,result,color='red')  #x에 값을 기준으로 선을 찍는다. 
plt.show()
#데이터끼리의 오차범위가 가장 적은 이차함수


