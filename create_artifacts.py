import cv2
import random
import numpy as np
import os
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image


def load_data(source_dir, img_size, training_size, validation_size, testing_size):
    male_train_filenames = [f for f in os.listdir(source_dir + "Train/Male")]
    female_train_filenames = [f for f in os.listdir(source_dir + "Train/Female")]

    male_valid_filenames = [f for f in os.listdir(source_dir + "Valid/Male")]
    female_valid_filenames = [f for f in os.listdir(source_dir + "Valid/Female")]

    male_test_filenames = [f for f in os.listdir(source_dir + "Test/Male")]
    female_test_filenames = [f for f in os.listdir(source_dir + "Test/Female")]

    rand_male_train = random.sample(male_train_filenames, training_size // 2)
    rand_female_train = random.sample(female_train_filenames, training_size // 2)
    rand_male_valid = random.sample(male_valid_filenames, validation_size // 2)
    rand_female_valid = random.sample(female_valid_filenames, validation_size // 2)
    rand_male_test = random.sample(male_test_filenames, testing_size // 2)
    rand_female_test = random.sample(female_test_filenames, testing_size // 2)

    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    test_images = []
    test_labels = []

    for image_male in rand_male_train:
        image_path = os.path.join(source_dir + "Train/Male", image_male)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        train_images.append(image)
        train_labels.append(0)
        
    for image_female in rand_female_train:
        image_path = os.path.join(source_dir + "Train/Female", image_female)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        train_images.append(image)
        train_labels.append(1)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)  
    
    for image_male in rand_male_valid:
        image_path = os.path.join(source_dir + "Valid/Male", image_male)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        valid_images.append(image)
        valid_labels.append(0)
        
    for image_female in rand_female_valid:
        image_path = os.path.join(source_dir + "Valid/Female", image_female)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        valid_images.append(image)
        valid_labels.append(1)
        
    valid_images = np.array(valid_images)
    valid_labels = np.array(valid_labels)  
        
    for image_male in rand_male_test:
        image_path = os.path.join(source_dir + "Test/Male", image_male)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        test_images.append(image)
        test_labels.append(0)
        
    for image_female in rand_female_test:
        image_path = os.path.join(source_dir + "Test/Female", image_female)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        test_images.append(image)
        test_labels.append(1)
        
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
        
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels


def load_data_v1(source_dir, img_size, training_size, validation_size, testing_size):
    male_train_filenames = [f for f in os.listdir(source_dir + "Train/Male")]
    female_train_filenames = [f for f in os.listdir(source_dir + "Train/Female")]

    male_valid_filenames = [f for f in os.listdir(source_dir + "Valid/Male")]
    female_valid_filenames = [f for f in os.listdir(source_dir + "Valid/Female")]

    male_test_filenames = [f for f in os.listdir(source_dir + "Test/Male")]
    female_test_filenames = [f for f in os.listdir(source_dir + "Test/Female")]

    rand_male_train = random.sample(male_train_filenames, training_size // 2)
    rand_female_train = random.sample(female_train_filenames, training_size // 2)
    rand_male_valid = random.sample(male_valid_filenames, validation_size // 2)
    rand_female_valid = random.sample(female_valid_filenames, validation_size // 2)
    rand_male_test = random.sample(male_test_filenames, testing_size // 2)
    rand_female_test = random.sample(female_test_filenames, testing_size // 2)

    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    test_images = []
    test_labels = []
    predict_train_images  = []
    
    for image_male in rand_male_train:
        image_path = os.path.join(source_dir + "Train/Male", image_male)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        train_images.append(image)
        train_labels.append(0)
        image = np.reshape(image, (1, 224, 224, 3))
        predict_train_images.append(image)

        
    for image_female in rand_female_train:
        image_path = os.path.join(source_dir + "Train/Female", image_female)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        train_images.append(image)
        train_labels.append(1)
        image = np.reshape(image, (1, 224, 224, 3))
        predict_train_images.append(image)
        
    for image_male in rand_male_valid:
        image_path = os.path.join(source_dir + "Valid/Male", image_male)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        valid_images.append(image)
        valid_labels.append(0)
        
    for image_female in rand_female_valid:
        image_path = os.path.join(source_dir + "Valid/Female", image_female)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        valid_images.append(image)
        valid_labels.append(1)
        
    valid_images = np.array(valid_images)
    valid_labels = np.array(valid_labels)  
        
    for image_male in rand_male_test:
        image_path = os.path.join(source_dir + "Test/Male", image_male)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        test_images.append(image)
        test_labels.append(0)
        
    for image_female in rand_female_test:
        image_path = os.path.join(source_dir + "Test/Female", image_female)
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        test_images.append(image)
        test_labels.append(1)
        
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
   
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels, predict_train_images
