import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils, layers, datasets, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

import numpy as np
import matplotlib.pyplot as plt

cifar_mnist = datasets.cifar10
fashion_mnist = datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = cifar_mnist.load_data()

print(x_train.shape, y_train.shape)

# Data Labeling
class_names = [
    'Airplane',
    'Car',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

# data visualization
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

#Hyper Parameters & Preprocessing
batch_size = 64
num_classes = 10
epochs = 3

x_train = x_train.astype('float32')
x_train = x_train/255

x_test = x_test.astype('float32')
x_test = x_test.astype('float32')

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

def plt_show_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.show()


def plt_show_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.show()
    

def CNN_model(x_train, y_train, x_test, y_test):
    model = keras.Sequential([
        Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=x_train.shape[1:], activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(num_classes, activation=tf.nn.softmax)
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=0.01),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping]
    )

    loss, acc = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)



    print("\nLoss: {}, Acc: {}".format(loss, acc))

    return history

history = CNN_model(x_train, y_train, x_test, y_test)