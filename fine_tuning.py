import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import scipy
from tensorflow.keras.optimizers import Adam

def custom_loss(y_true, y_pred, mse):
    alpha = 1
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + alpha * mse
    
def FineTune(model, train_images, train_labels, val_images, val_labels, test_images, test_labels, batch_size, filename):
        
    history = model.fit(
        train_images,
        train_labels,
        batch_size = batch_size,
        steps_per_epoch = len(train_images) // batch_size,
        validation_data = (val_images, val_labels),
        epochs=3
    )

    scores = model.evaluate(test_images, test_labels)
    print("Test Loss:", scores[0])
    print("Test Accuracy:", scores[1])

    model.save_weights(filename)
    return model


def eval(model, test_images, test_labels):
    return model.evaluate(test_images, test_labels)
    
    
def create_model(x_model):
    for layer in x_model.layers:
        layer.trainable = False

    #Add new classifier layers on top of the ResNet50 base layers
    x = Flatten()(x_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    # x = x_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    # predicitions = Dense(1, activation='softmax')(x)
        
    model = Model(inputs = x_model.input, outputs = x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    return model


def create_algorithm_model(x_model, mse):
    for layer in x_model.layers:
        layer.trainable = False

    #Add new classifier layers on top of the ResNet50 base layers
    x = Flatten()(x_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs = x_model.input, outputs = x)
    model.compile(loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, mse), optimizer='adam', metrics=['accuracy'])
    return model