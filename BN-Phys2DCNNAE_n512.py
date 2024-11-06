# 01_BN-2DAE.py
# 2024 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Authors:
# Kai Fukami and Kunihiko Taira 
## We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
#     Ref: K. Fukami and K. Taira,
#     “Seeking universal coordinates for multi-source turbulent flow data,”
#     in review, 2024
#
# The code is written for educational clarity and not for speed.
# -- version 1: Aug 19, 2024


#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM, Concatenate, BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm as tqdm
from scipy.io import loadmat
import mat73
from keras.layers import LeakyReLU


#import tensorflow._api.v2.compat.v1 as tf

#tf.disable_v2_behavior()

input_img = Input(shape=(128,128,1))
act = 'tanh'

x1 = Conv2D(32, (3,3),activation=act, padding='same')(input_img)
x1 = BatchNormalization()(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1) #64, 64
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1) #32, 32
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1) #16, 16
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)

x_lat = Conv2D(2, (3,3),activation=act, padding='same')(x1)

x_CL = Reshape([16*16*2])(x_lat)
x_CL = Dense(256,activation=act)(x_CL)
x_CL = Dense(128,activation=act)(x_CL)
x_CL = Dense(32,activation=act)(x_CL)
x_CL_final = Dense(1)(x_CL)


x1 = Conv2D(8, (3,3),activation=act, padding='same')(x_lat)
x1 = BatchNormalization()(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = BatchNormalization()(x1)


x_final = Conv2D(1, (3,3),padding='same')(x1)

autoencoder = Model(input_img, [x_final,x_CL_final])

from keras.losses import binary_crossentropy, mse
autoencoder.compile(optimizer='adam', loss='mse',loss_weights=[1,0.01])



# y_Train_ALL and y_CL need to be prepared.


from keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, X_train1, X_test1 = train_test_split(y_Train_ALL,y_CL, test_size=0.2, random_state=None)
del y_1, y_Train_ALL, y_CL


model_cb=ModelCheckpoint('./Model/model.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
history = autoencoder.fit(X_train,[X_train,X_train1],epochs=500000,batch_size=16,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, [X_test,X_test1]))
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./History/history.csv',index=False)






