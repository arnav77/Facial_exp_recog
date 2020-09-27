#psuedo code for recognition model
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

num_classes = 7
img_rows, img_cols = 48,48
batch_size = 8

train_data = 'C:\Users\HP\Downloads\Facial exp recog\train'
validation_data = 'C:\Users\HP\Downloads\Facial exp recog\validation'

train_data_gen= ImageDataGenerator(rescale=1./255, rotation=30, shear_range=0.3, 

)