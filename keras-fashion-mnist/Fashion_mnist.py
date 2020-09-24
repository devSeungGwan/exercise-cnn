from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# Load Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
y_train = to_categorical(y_train.astype('float32'), 10)

x_test = x_test.astype('float32') / 255
y_test = to_categorical(y_test.astype('float32'), 10)


print(x_train.shape, y_train.shape)

# Show Data
# plt.figure(figsize=(6, 6))
# for i in range(36):
#     plt.subplot(6, 6, i+1)
#     plt.imshow(x_train[i], cmap=plt.cm.gray)
#     plt.axis("off")

# plt.show()

def fashion_model():
    model = Sequential([
        Conv2D(input_shape=x_train.shape[1:], filters=50, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])

    model.summary()

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = Adam(lr=0.001),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=25, 
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping]
    )

    loss, acc = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)

    print("\nLoss: {}, Acc: {}".format(loss, acc))


fashion_model()