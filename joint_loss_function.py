import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import cv2
import numpy as np
import random
import os

pertubations = ["color-jitter", "horizontal-flippping"]

def pertubate(image_path):

    img = cv2.imread(image_path)
    order = random.randint(0, 1)
    if order == 0:
        # Define the random color jitter parameters
        brightness = np.random.randint(-30, 30)
        contrast = np.random.uniform(0.5, 1.5)
        saturation = np.random.uniform(0.5, 1.5)
        hue = np.random.randint(-10, 10)

        # Convert image to HSV color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply brightness, contrast, and saturation adjustments to the image
        img_hsv[:,:,2] = cv2.addWeighted(img_hsv[:,:,2], contrast, 0, brightness, 0)
        img_hsv[:,:,1] = cv2.addWeighted(img_hsv[:,:,1], saturation, 0, 0, 0)

        # Convert the image back to BGR color space
        jittered_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return jittered_img
    
    if order == 1:
        flipped_img = np.fliplr(img)
        return flipped_img
    
    
    # Show the resulting image
    # cv2.imshow('Original image', img)
    # cv2.imshow('jittered image', jittered_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def pick_generalizable_images(source_dir, generalizable_size, img_size):
    
    male_filenames = [f for f in os.listdir(source_dir + "Train/Male")]
    female_filenames = [f for f in os.listdir(source_dir + "Train/Female")]
    
    rand_male = random.sample(male_filenames, generalizable_size // 2)
    rand_female = random.sample(female_filenames, generalizable_size // 2)
    
    original_images = []
    pertubated_images = []
    
    for image_male in rand_male:
        image_path = os.path.join(source_dir + "Train/Male", image_male)
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        image = np.reshape(image, (1, 224, 224, 3))
        original_images.append(image)
        
        pertubated_image = pertubate(image_path)
        pertubated_image = cv2.resize(pertubated_image, img_size)
        pertubated_image = np.array(pertubated_image, dtype = np.float32) / 255.0
        pertubated_image = np.reshape(pertubated_image, (1, 224, 224, 3))
        pertubated_images.append(pertubated_image)
        
    for image_female in rand_female:
        image_path = os.path.join(source_dir + "Train/Female", image_female)
             
        image = cv2.imread(image_path)
        image = cv2.resize(image, img_size)
        image = np.array(image, dtype = np.float32) / 255.0
        image = np.reshape(image, (1, 224, 224, 3))
        original_images.append(image)
        
        pertubated_image = pertubate(image_path)
        pertubated_image = cv2.resize(pertubated_image, img_size)
        pertubated_image = np.array(pertubated_image, dtype = np.float32) / 255.0
        pertubated_image = np.reshape(pertubated_image, (1, 224, 224, 3))
        pertubated_images.append(pertubated_image)
        
    return original_images, pertubated_images


def calculate_mse(model, source_dir, generalizable_size, img_size):
    
    original_images, pertubated_images = pick_generalizable_images(source_dir, generalizable_size, img_size)
    mean_square_error = 0
    
    for index, original_image in enumerate(original_images):
        pertubated_image = pertubated_images[index]
        original_prediction = model.predict(original_image)
        pertubated_prediction = model.predict(pertubated_image)
        if (original_prediction > 0.5 and pertubated_prediction > 0.5) or (original_prediction < 0.5 and pertubated_prediction < 0.5):
            mean_square_error += 0
        else:
            mean_square_error += 1
    mean_square_error = mean_square_error / generalizable_size
    return mean_square_error
    