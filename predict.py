import cv2
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

import os
import shutil

IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic')

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
model.load(MODEL_NAME)


def predict_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data = img.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    plt.imshow(img, cmap='gray')
    plt.title(str_label)
    plt.show()


def organize_images(src_folder, dst_folder):
    # Create the Cat and Dog folders if they don't exist
    os.makedirs(os.path.join(dst_folder, 'Cat'), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, 'Dog'), exist_ok=True)

    # Iterate through all images in the source folder
    for img_name in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_name)

        # Check if the file is an image
        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Predict the label using the previously defined function
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data = img.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        label = 'Dog' if np.argmax(model_out) == 1 else 'Cat'

        # Move the image to the appropriate folder
        dst_img_path = os.path.join(dst_folder, label, img_name)
        shutil.move(img_path, dst_img_path)

# Example usage:
image_path = r'example.jpg'
predict_single_image(image_path)


image_path = r'th-2217569998.jpg'
predict_single_image(image_path)


src_folder = r'C:\Users\Barbruh\PycharmProjects\ImageClassifierUsingCNN\Test'
dst_folder = r'C:\Users\Barbruh\PycharmProjects\ImageClassifierUsingCNN\Result'
organize_images(src_folder, dst_folder)