# Constants
WIDTH   = 80
HEIGHT  = 60

# Commented out IPython magic to ensure Python compatibility.
# Imports
# %tensorflow_version 1.x
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
import random
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Reshape, Input, ZeroPadding2D, GlobalMaxPool2D

# Data loading
data=np.load("/content/drive/MyDrive/BeamNG traindata Grascale/training_data_v2.npy",allow_pickle=1)

# Splitting data into training and testing
train   = data[:-round(len(data)*0.20)]
test    = data[-round(len(data)*0.20):]


X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = np.array([i[1] for i in test])

def nvidia_modified():
  model = Sequential()
  model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(WIDTH, HEIGHT,1),activation='elu',data_format='channels_last'))
  model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(48, kernel_size=(5,5),  activation='elu'))
  model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
  model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation='elu'))
  model.add(Dense(9))
  model.compile(Adam(lr=0.001), loss = 'mse', metrics = ['accuracy'])
  
  return model
model = nvidia_modified()
print (model.summary())

def batch_generator(train_data_to_generate, batch_size):

  while True:
    batch_img = []
    batch_ch = []

    for i in range(batch_size):
      random_index = random.randint(0, len(train_data_to_generate)-1)
      image, choice = train_data_to_generate[random_index]
      im = image
      ch = choice
      im = im.reshape(WIDTH,HEIGHT,1)
      batch_img.append(im)
      batch_ch.append(ch)

    yield(np.asarray(batch_img), np.asarray(batch_ch))

history = model.fit_generator(batch_generator(train,300),
steps_per_epoch = 150,
epochs = 25,
validation_data = batch_generator(test, 100),
validation_steps = 50,
verbose = 1, 
shuffle=1)

plt.rcParams["figure.figsize"] = (20,10)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

model.save("beamboy_nvidia.h5")

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_x, test_y, batch_size=10)
print("test loss, test acc:", results)