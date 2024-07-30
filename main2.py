import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#main_dir = "C:/Users/Shree/Downloads/Fruit-Classifier-main/Fruit-Classifier-main/Data2"

main_dir = r"C:\Users\flash\Desktop\(Part2)Grape Leaf Disease Detection_Working_12 JAN\Grape Leaf Disease Detection_Working_12 JAN\Data"
os.listdir(main_dir)

test_dir = main_dir + '/test'
os.listdir(test_dir)

train_dir = main_dir + '/train'
os.listdir(train_dir)

# train set

# Creating a data generator
train = ImageDataGenerator(
  rescale=1./255,
    horizontal_flip = True,
    vertical_flip = True,
    validation_split = 0.2
)
train_datagen = train.flow_from_directory(
        train_dir,
        batch_size=32,
        target_size=(128,128),
        class_mode='sparse',
        subset = 'training')

val_datagen = train.flow_from_directory(
        train_dir,
        batch_size=32,
        target_size=(128,128),
        class_mode='sparse',
        subset = 'validation')

# Printing the training set
train_labels = (train_datagen.class_indices)
print(train_labels)

# test set

# Create a data generator
test_datagen = ImageDataGenerator(
  rescale=1./255,
    horizontal_flip = True,
    vertical_flip = True
)
test_datagen = test_datagen.flow_from_directory(
        test_dir,
        batch_size=32,
        target_size=(128,128),
        class_mode='sparse')

# Printing the test set
test_labels = (test_datagen.class_indices)
print(test_labels)

# image configuration

for image_batch, label_batch in train_datagen:
    break
image_batch.shape, label_batch.shape

# This would be same for test set


# build the model

model=Sequential()

#Convolution blocks
model.add(Conv2D(32, kernel_size = (3,3),
                 padding='same',
                 input_shape=(128,128,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64, kernel_size = (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, kernel_size = (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))

#Classification layers
model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(5,activation='softmax'))


model.summary()

# compile the model

model.compile(optimizer = 'Adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# training the model
trained_model = model.fit(train_datagen, epochs=25,
                          batch_size = 32,
                          validation_data = val_datagen, verbose = 1)

#model_dir = 'C:/Users/Admin/Downloads/Fruit-Classifier-main/Fruit-Classifier-main/Model/model2.h5'
model_dir = r"C:\Users\flash\Desktop\(Part2)Grape Leaf Disease Detection_Working_12 JAN\Grape Leaf Disease Detection_Working_12 JAN\Data"

model.save(model_dir)

from tensorflow import keras
model = keras.models.load_model(model_dir)

# Testing the model
