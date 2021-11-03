
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

    

# getting data
#>>>directories
base_dir = r'C:\Users\Mahesh B\OneDrive\Desktop\Minor project\dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_diseasedleaf = os.path.join(train_dir, 'diseased cotton leaf')
train_diseasedplant = os.path.join(train_dir, 'diseased cotton plant')
train_plant = os.path.join(train_dir, 'fresh cotton plant')
train_leaf= os.path.join(train_dir, 'fresh cotton leaf')

test_diseasedleaf = os.path.join(test_dir, 'diseased cotton leaf')
test_diseasedplant = os.path.join(test_dir, 'diseased cotton plant')
test_plant = os.path.join(test_dir, 'fresh cotton plant')
test_leaf= os.path.join(test_dir, 'fresh cotton leaf')

#>>>No of images
num_diseasedleaf = len(os.listdir(train_diseasedleaf))
num_diseasedplant = len(os.listdir(train_diseasedplant))
num_plant = len(os.listdir(train_plant))
num_leaf = len(os.listdir(train_leaf))

num_diseasedleaf2 = len(os.listdir(test_diseasedleaf))
num_diseasedplant2 = len(os.listdir(test_diseasedplant))
num_plant2 = len(os.listdir(test_plant))
num_leaf2 = len(os.listdir(test_leaf))


total_train = num_diseasedleaf + num_diseasedplant + num_plant + num_leaf
total_val = num_diseasedleaf2 + num_diseasedplant2 + num_plant2 + num_leaf2

BATCH_SIZE = 8
IMG_SHAPE = 224


#generators

#prevent memorization
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_image_generator = ImageDataGenerator(
    rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=test_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')


# model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
model =Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)), # RGB
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(96, (3,3), activation='relu'), 
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(96, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    
    Dense(7, activation='softmax')
    
    ])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


EPOCHS = 10

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )

model.save("model.h5")
# analysis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()











