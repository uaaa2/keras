# 0차원 1 "스칼라"
# 1차원 [1,2,3] "벡터"
# 2차원 [2차원리스트] '메트릭스'
# 3차원 [3차원리스트] 'Tensor'
# 4차원 [3차원리스트 여러개] 'Tensor'


import numpy as np

x1 = np.array([1,2,3])             #x1 = (3,) 벡터
print("x1 = ",x1.shape) 

x2 = np.array([                     #x2 = (1, 3) 메트릭스 
               [1,2,3]
              ])    
print("x2 =",x2.shape)


x3 = np.array([                    #x3 = (2, 3) 메트릭스
               [1,2,3],
               [4,5,6]
              ])    
print("x3 =",x3.shape)

x4 = np.array([[1,2],[3,4]])        #x4 = (2, 2)
print(f"x4 = {x4.shape}")

x5 = np.array([[[[1]]],[[[2]]]])    #x5 = (2, 1, 1, 1)
print(f"x5 = {x5.shape}")