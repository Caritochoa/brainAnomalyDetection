import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
from os import listdir
import glob
from os.path import splitext, basename
import cv2
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

#show the data set

def preprocess_image(image_path,resize=False):    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


# Create a list of image paths and show the data set
image_paths = glob.glob(r'C:\Users\carit\Desktop\tumorDetection\Brain-Tumor-Detection\yesno\yes/*.jpg')
print("Found %i images..."%(len(image_paths)))

# Visualize data in subplot to check
fig = plt.figure(figsize=(12,8))
cols = 5
rows = 4
fig_list = []
for i in range(cols*rows):
    fig_list.append(fig.add_subplot(rows,cols,i+1))
    title = splitext(basename(image_paths[i]))[0]
    fig_list[-1].set_title(title)
    img = preprocess_image(image_paths[i],True)
    plt.axis(False)
    plt.imshow(img)

plt.tight_layout(True)
plt.show()

# Create a list of image paths and show the data set
image_paths = glob.glob(r'C:\Users\carit\Desktop\tumorDetection\Brain-Tumor-Detection\yesno\no/*.jpg')
print("Found %i images..."%(len(image_paths)))

# Visualize data in subplot 
fig = plt.figure(figsize=(12,8))
cols = 5
rows = 4
fig_list = []
for i in range(cols*rows):
    fig_list.append(fig.add_subplot(rows,cols,i+1))
    title = splitext(basename(image_paths[i]))[0]
    fig_list[-1].set_title(title)
    img = preprocess_image(image_paths[i],True)
    plt.axis(False)
    plt.imshow(img)

plt.tight_layout(True)
plt.show() 

#Directories images
print(len(os.listdir(r'C:\Users\carit\Desktop\tumorDetection\Brain-Tumor-Detection\yesno\yes')))
print(len(os.listdir(r'C:\Users\carit\Desktop\tumorDetection\Brain-Tumor-Detection\yesno\no')))


#creation of this dirs after they are processed and augmented  are kept in this dirs
try:
    os.mkdir('trial1')
    os.mkdir('trial1/augmented data1/')
    os.mkdir('trial1/augmented data1/training')
    os.mkdir('trial1/augmented data1/training/yes1')
    os.mkdir('trial1/augmented data1/training/no1')
    os.mkdir('trial1/augmented data1/testing')
    os.mkdir('trial1/augmented data1/testing/yes1')
    os.mkdir('trial1/augmented data1/testing/no1')
    os.mkdir('trial1/augmented data1/yesreal')
    os.mkdir('trial1/augmented data1/noreal')
except OSError:
    pass

    def augment_data(file_dir, n_generated_samples, save_to_dir):
    #from keras.preprocessing.image import ImageDataGenerator 
    #from os import listdir changes perspective to be analised
    
    # generate lots or ensro image data with real time augmentation
    # agumentation of the data es transform it to generate more 
    
    data_gen = ImageDataGenerator(rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True, 
                                  vertical_flip=True, 
                                  fill_mode='nearest'
                                 )
    
    for filename in listdir(file_dir):
        # load the image
        image = cv2.imread(file_dir + '\\' + filename)
        # reshape the image
        image = image.reshape((1,)+image.shape)
        # prefix of the names for the generated sampels.
        save_prefix = 'aug_' + filename[:-4]
        # generate 'n_generated_samples' sample images
        i=0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, 
                                           save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples:
                break


augmented_data_path = 'trial1/augmented data1/' 
# augmented  data for the examples with label equal to 'yes' representing tumurous examples
augment_data(file_dir=r'C:\Users\carit\Desktop\tumorDetection\Brain-Tumor-Detection\yesno\yes', n_generated_samples=6, save_to_dir=augmented_data_path+'yesreal')
# augmented data for the examples with label equal to 'no' representing non-tumurous examples
augment_data(file_dir=r'C:\Users\carit\Desktop\tumorDetection\Brain-Tumor-Detection\yesno\no', n_generated_samples=9, save_to_dir=augmented_data_path+'noreal')

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []
    
    for unitData in os.listdir(SOURCE):
        data = SOURCE + unitData
        if(os.path.getsize(data) > 0):
            dataset.append(unitData)
        else:
            print('Skipped ' + unitData)
            print('Invalid file i.e zero size')
    
    train_set_length = int(len(dataset) * SPLIT_SIZE)
    test_set_length = int(len(dataset) - train_set_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = dataset[0:train_set_length]
    test_set = dataset[-test_set_length:]
       
    for unitData in train_set:
        temp_train_set = SOURCE + unitData
        final_train_set = TRAINING + unitData
        copyfile(temp_train_set, final_train_set)
    
    for unitData in test_set:
        temp_test_set = SOURCE + unitData
        final_test_set = TESTING + unitData
        copyfile(temp_test_set, final_test_set)
        
        
YES_SOURCE_DIR = "trial1/augmented data1/yesreal/"
TRAINING_YES_DIR = "trial1/augmented data1/training/yes1/"
TESTING_YES_DIR = "trial1/augmented data1/testing/yes1/"
NO_SOURCE_DIR = "trial1/augmented data1/noreal/"
TRAINING_NO_DIR = "trial1/augmented data1/training/no1/"
TESTING_NO_DIR = "trial1/augmented data1/testing/no1/"
split_size = .8
split_data(YES_SOURCE_DIR, TRAINING_YES_DIR, TESTING_YES_DIR, split_size)
split_data(NO_SOURCE_DIR, TRAINING_NO_DIR, TESTING_NO_DIR, split_size)


print(len(os.listdir('trial1/augmented data1/training/yes1')))
print(len(os.listdir('trial1/augmented data1/testing/yes1')))
print(len(os.listdir('trial1/augmented data1/training/no1')))
print(len(os.listdir('trial1/augmented data1/testing/no1')))        

#Creation of the model
model = tf.keras.models.Sequential([
    #the conv2 Laver creates a kernel,  the first params are filters that the covolutional layer will learn, secon param kernel area
    # a commom practice is to raise the filter in each layer of conv2d using de maxpooling to reduce the spatial dimensions. 
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = "trial1/augmented data1/training"
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    batch_size=10, 
                                                    class_mode='binary', 
                                                    target_size=(150, 150))

VALIDATION_DIR = "trial1/augmented data1/testing"
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size=10, 
                                                         class_mode='binary', 
                                                         target_size=(150, 150))

history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator)

#getting accuracy values
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs


plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.title('Training and validation loss')

#testing the model in a simple method.

from keras.preprocessing import image
import cv2
import os
import glob
img_dir = r'C:\Users\carit\Desktop\test_img'
#img_dir = 'trial1/augmented data1/training/no1/'
#img_dir = 'trial1/augmented data1/training/yes1/'
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
result = []
for f1 in files:
    test_image = image.load_img(f1, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    images = np.vstack([test_image])
    classes = model.predict(images, batch_size=10)
    classes = np.round(classes)
    data.append(f1)
    result.append(classes)
    
#for i in result: 
    if i == [[0.]]:
        print ("no")
    else:
        print ("yes")
    


                              


