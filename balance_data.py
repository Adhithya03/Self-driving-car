import numpy as np
import pandas as pd
from collections import Counter

train_data = np.load('training-data\\training_data-full.npy',allow_pickle=1)
# s   80700    =  [1,0,0,0,0,0,0,0,0]
# nk  50240    =  [0,0,0,0,0,0,0,0,1]
# sr  32788    =  [0,0,0,0,0,1,0,0,0]
# sl  32214    =  [0,0,0,0,1,0,0,0,0]
# b   4558     =  [0,1,0,0,0,0,0,0,0] 

print(f"Before removing data : {len(train_data)}")

straight        =[]
straight_left   =[]
straight_right  =[]
no_keys         =[]
brake           =[]

for data in train_data:
    img    = data[0]
    choice = data[1]

    if   choice==[1,0,0,0,0,0,0,0,0]:
        
        straight.append([img,choice])

    elif choice==[0,0,0,0,0,0,0,0,1]:
        
        no_keys.append([img,choice])

    elif choice==[0,0,0,0,0,1,0,0,0]:
        
        straight_right.append([img,choice])

    elif choice==[0,0,0,0,1,0,0,0,0]:
        
        straight_left.append([img,choice])

    elif choice==[0,1,0,0,0,0,0,0,0]:
        brake.append([img,choice])

    else:
        print("NO mathces")

straight=straight[:len(straight_left)]
no_keys =no_keys[:len(straight_left)]
straight_right=straight_right[:len(straight_left)]
print(len(straight))
print(len(straight_left))
print(len(straight_right))
print(len(no_keys))
print(len(brake))

final_data = straight + no_keys + straight_right + straight_left + brake

# np.save("training-data\\training_data_v2.npy" , final_data)