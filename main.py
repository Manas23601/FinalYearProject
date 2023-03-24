import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
import scipy
import random
import os
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import create_artifacts
import fine_tuning
import joint_loss_function


# 0 is Male, 1 is Female
labels = ["Male", "Female"]
directories = ["Train", "Valid", "Test"]
source_dir = "./Working/"

def find_confident_samples(model, predict_train_images, train_images):
    confident_images = []
    confident_labels = []
    for index, image in enumerate(predict_train_images):
        prediction = model.predict(image)
        if prediction < 0.1 or prediction > 0.9:
            confident_images.append(train_images[index])
            if prediction < 0.1:
                confident_labels.append(0)
            else:
                confident_labels.append(1)
    print(len(confident_labels))

    confident_images = np.array(confident_images)
    confident_labels = np.array(confident_labels)
    return confident_images, confident_labels
    
validation_size = 50
testing_size = 800
img_size = (224, 224)
num_of_classes = 2
batch_size = 32

# Co-Teaching Parameters
co_teaching_batch_sizes = [100, 250]
confidence_threshold = .90
num_of_iterations = 1

# HUL Algorithm Parameters
generalizable_sizes = [100, 250]


results_1 = []
results_2 = []

file_names_1 = ['./models/fine_tuned_vgg1_100.h5', './models/fine_tuned_vgg1_200.h5']
file_names_2 = ['./models/fine_tuned_vgg2_100.h5', './models/fine_tuned_vgg2_200.h5']
init_fine_tuned = [100, 200]

for index_l in range(len(file_names_1)):
    
    file_name_1 = file_names_1[index_l]
    file_name_2 = file_names_2[index_l]
    
    for index_i in range(len(co_teaching_batch_sizes)):
        for index_j in range(len(generalizable_sizes)):
            for index_k in range(num_of_iterations):
            
                # Hyperparameters for albation study
                self_training_batch_size = co_teaching_batch_sizes[index_i]
                generalizable_size = generalizable_sizes[index_j]
                print("Initially fine tuned with : ", init_fine_tuned[index_l])
                print("Co-Teaching Batch Size: ", self_training_batch_size)
                print("HUL Batch Size : ", generalizable_size)
                print("Co-Teaching Iteration number : ", index_k + 1)
                print("Loading Weights : ", file_name_1, file_name_2)
                
                # Create two VGG models
                vgg1 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                vgg2 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                
                # Define model
                vgg1 = fine_tuning.create_model(vgg1)
                vgg2 = fine_tuning.create_model(vgg2)

                # Load Weights
                vgg1.load_weights(file_name_1)
                vgg2.load_weights(file_name_2)
                
                # Calculate Mean Square Error Loss for HUL
                vgg1_mse = joint_loss_function.calculate_mse(vgg1, source_dir, generalizable_size, img_size)
                vgg2_mse = joint_loss_function.calculate_mse(vgg2, source_dir, generalizable_size, img_size)

                # Harnessing Unlabelled Data
                
                # Create two VGG models
                vgg1 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                vgg2 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                
                # Define the models which supports custome loss function
                vgg1 = fine_tuning.create_algorithm_model(vgg1, vgg1_mse)
                vgg2 = fine_tuning.create_algorithm_model(vgg2, vgg2_mse)
                
                # Load Weights
                vgg1.load_weights('./models/fine_tuned_vgg1_200.h5')
                vgg2.load_weights('./models/fine_tuned_vgg2_200.h5')
                
                # Load Separate Data
                train_images1, train_labels1, valid_images1, valid_labels1, test_images1, test_labels1, predict_train_images1 = create_artifacts.load_data_v1(source_dir, img_size, self_training_batch_size, validation_size, testing_size)
                train_images2, train_labels2, valid_images2, valid_labels2, test_images2, test_labels2, predict_train_images2 = create_artifacts.load_data_v1(source_dir, img_size, self_training_batch_size, validation_size, testing_size)
                
                # Find Confident Samples
                vgg1_confident_images, vgg1_confident_labels = find_confident_samples(vgg1, predict_train_images1, train_images1)
                vgg2_confident_images, vgg2_confident_labels = find_confident_samples(vgg2, predict_train_images2, train_images2)
                
                # Perform Fine-Tuning
                file_name_1 = "./models/hul_vgg1_f" + str(init_fine_tuned[index_l])  + "_c" + str(self_training_batch_size) +  "_g" + str(generalizable_size) +  "_cti" + str(index_k + 1) + ".h5"
                vgg1 = fine_tuning.FineTune(vgg1, vgg2_confident_images, vgg2_confident_labels, valid_images2, valid_labels2, test_images2, test_labels2, batch_size, file_name_1)
                results_1 = fine_tuning.eval(vgg1, test_images1, test_labels1)
                file_name_2 = "./models/hul_vgg2_f" + str(init_fine_tuned[index_l]) + "_c" + str(self_training_batch_size) +  "_g" + str(generalizable_size) + "_cti" + str(index_k + 1) + ".h5"
                vgg2 = fine_tuning.FineTune(vgg2, vgg1_confident_images, vgg1_confident_labels, valid_images1, valid_labels1, test_images1, test_labels1, batch_size, file_name_2)
                results_2 = fine_tuning.eval(vgg2, test_images2, test_labels2)
                
                print("Accuracy Result : ", max(results_1, results_2))
        
# print("Printing Results")
# print("Results for Model-1")
# for l in range(len(file_name_1)):
#     print("Initially Fine tuned with : ", init_fine_tuned[l])
#     for i in range(len(co_teaching_batch_sizes)):
#         for j in range(len(generalizable_sizes)):
#             for k in range(num_of_iterations):
#                 print("Co-Teaching Batch Size : ", co_teaching_batch_sizes[i], "Generalizable_size : ", generalizable_sizes[j], "Iteration number : ", k+1,  "Accuracy : ", results_1[i])
    
# print("Results for Model-2")
# for l in range(len(file_name_1)):
#     print("Initially Fine tuned with : ", init_fine_tuned[l])
#     for i in range(len(co_teaching_batch_sizes)):
#         for j in range(len(generalizable_sizes)):
#             for k in range(num_of_iterations):
#                 print("Co-Teaching Batch Size : ", co_teaching_batch_sizes[i], "Generalizable_size : ", generalizable_sizes[j], "Iteration number : ", k+1,  "Accuracy : ", results_2[i])