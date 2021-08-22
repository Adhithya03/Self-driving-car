import win32api as wapi,os,cv2,numpy as np,time
from grabscreen import grab_screen
keyList =  [0x25,0x26,0x27,0x28]

s       =  [1,0,0,0,0,0,0,0,0]
b       =  [0,1,0,0,0,0,0,0,0]
l       =  [0,0,1,0,0,0,0,0,0]
r       =  [0,0,0,1,0,0,0,0,0]
sl      =  [0,0,0,0,1,0,0,0,0]
sr      =  [0,0,0,0,0,1,0,0,0]
bl      =  [0,0,0,0,0,0,1,0,0]
br      =  [0,0,0,0,0,0,0,1,0]
nk      =  [0,0,0,0,0,0,0,0,1]

straight    = 38
left        = 37
right       = 39
brake       = 40

def key_check():
    keys=[]
    for key in keyList:
        if wapi.GetAsyncKeyState(key):
            keys.append(key)
    return keys

def keys_to_output(keys):

    if straight and left in keys:
        output = sl
    elif straight and right in keys:
        output = sr
    elif brake and left in keys:
        output = bl
    elif brake and right in keys:
        output = br
    elif straight in keys:
        output = s
    elif brake in keys:
        output = b
    elif left in keys:
        output = l
    elif right in keys:
        output = r
    else:
        output = nk
    return output

def countdown(wait):
    for i in range(wait):
        time.sleep(1)
        print(f"T minus = {wait-i}")
        print("\n")

countdown(5)

training_data=[]
file_name = 'D:\\My-works\\dev\\Python\\Hobby projects\\Self-Driving-car\\training-data\\training_data-large.npy'

while True:

    if os.path.isfile(file_name):
        print('File exists, moving along')
    else:
        print('File does not exist, starting fresh!')        
        break

while 1:

        screen = grab_screen(region=(0,31,959,539))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,60))
        keys=key_check()
        output=keys_to_output(keys)
        training_data.append([screen,output])
        
        if len(training_data) % 1000 == 0:
            print(f"{len(training_data)} Common you can do it")
            np.save(file_name,training_data)
