import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.pooling import MaxPool2D

f = open("train.txt", "r")
f.readline()
train_images = []
train_labels = []

for line in f:
    train_image, train_label = line.split(',')
    img = PIL.Image.open('train+validation/' + train_image)
    train_images.append(np.array(img))
    train_labels.append(train_label)

train_images = np.array(train_images)
train_images = train_images / 255
train_labels = np.array(train_labels).astype('float')

f.close()
f = open("validation.txt", "r")
f.readline()

validation_images = []
validation_labels = []

for line in f:
    validation_image, validation_label = line.split(',')

    img = PIL.Image.open('train+validation/' + validation_image)
    validation_images.append(np.array(img))

    validation_labels.append(validation_label)

validation_images = np.array(validation_images)
validation_images = validation_images / 255
validation_labels = np.array(validation_labels).astype('float')

f.close()
f = open("test.txt", "r")
f.readline()
test_images = []
for line in f:
    if line[-1] == "\n":
        img = PIL.Image.open('test/' + line[:-1])
    else:
        img = PIL.Image.open('test/' + line)
    test_images.append(np.array(img))

test_images = np.array(test_images)
test_images = test_images / 255

cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size = (2, 2), activation = 'relu', input_shape = (16, 16, 3),
                        kernel_regularizer = keras.regularizers.l2(0.0015)),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, kernel_size=(2, 2), activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.0015)),

    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.35),

    keras.layers.Conv2D(128, kernel_size=(2, 2), activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.0015)),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(64, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.7),

    keras.layers.Dense(7, activation = 'softmax')
])

o = keras.optimizers.Adam(learning_rate=1e-5)
cnn_model.compile(optimizer = o, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
cnn_model.fit(train_images, train_labels, epochs = 200, batch_size = 32,
              validation_data = (validation_images, validation_labels))

rez = cnn_model.predict(test_images)

f.close()
f = open("test.txt")
f.readline()

g = open("test_cnn_documentatie.txt", "w")
g.write("id,label\n")

i = 0
for img_name in f:
    if img_name[-1] == "\n":
        g.write(img_name[:-1] + "," + str(rez[i].argmax()) + "\n")
    else:
        g.write(img_name + "," + str(rez[i].argmax()) + "\n")
    i = i + 1

f.close()
g.close()


