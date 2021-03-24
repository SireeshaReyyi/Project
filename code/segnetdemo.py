from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'C:/Users/imageNetToyDataset/train'
validation_data_dir = 'C:/Users/imageNetToyDataset/validation'

epochs = 5
nb_train_samples = 2000
nb_validation_samples = 50
batch_size = 16
input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

train_datagen = ImageDataGenerator(
rescale=1. / 255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='binary')

model.fit_generator(
train_generator,
steps_per_epoch=nb_train_samples // batch_size,
epochs=epochs,
validation_data=validation_generator,
validation_steps=nb_validation_samples // batch_size)

import numpy as np
import cv2
import csv
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, 
img_to_array, load_img
from scipy.misc import imresize
import scipy

def predict_labels(model):
    """writes test image labels and predictions to csv"""
    test_data_dir = "C:/Users/imageNetToyDataset/test"
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    shuffle=False,
    class_mode="binary")

    with open("prediction.csv", "w") as f:
        p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
        for _, _, imgs in os.walk(test_data_dir):
            print ("number of images: {}".format(len(imgs)))
            for im in imgs:
                print ("image:\n{}".format(im))
                pic_id = im.split(".")[0]
                imgPath = os.path.join(test_data_dir,im)
                print (imgPath)
                img = load_img(imgPath)
                img = imresize(img, size=(img_width, img_height))
                print ("img shape = {}".format(img.shape))

                test_x = img_to_array(img).reshape(3, img_width, img_height)

                print ("test_x shape = {}".format(test_x.shape))
                test_generator = test_datagen.flow(test_x,
                                               batch_size=1,
                                               shuffle=False)
                prediction = model.predict_generator(test_generator,1,epochs)
                p_writer.writerow([pic_id, prediction])

prediction=predict_labels(model)
