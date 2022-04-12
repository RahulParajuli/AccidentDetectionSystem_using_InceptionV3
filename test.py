from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import cv2
# import os
# import pickle

model = load_model('best_model.h5')
# model = pickle.load(open('C:/Users/batman/Desktop/Test/model.sav', 'rb'))

path = 'C:/Users/batman/Desktop/Test/Images/2676.png'

img = load_img(path, target_size = (256,256))
i = img_to_array(img)

i = preprocess_input(i)

input_arr= np.array([i])
input_arr.shape 
pred = np.argmax(model.predict(input_arr))


if pred ==0:
    print ("accident")
else:
    print("no accident")


image = cv2.imread(path)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()