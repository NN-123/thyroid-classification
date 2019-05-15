import tensorflow as tf
from tensorflow import keras
import numpy as np
import loader

data = loader.load()
data = list(filter(lambda x: x['tirads'] is not None, data))
labels = [x['tirads'] != '2' and x['tirads'] != '3' for x in data]
images = [x['image'] for x in data]

labels = np.array(labels)
images = np.array(images)

images = images.astype('float32') / 255
labels = keras.utils.to_categorical(labels, 2)

train_images = images[:-30, :, :, :]
test_images = images[-30:, :, :, :]

train_labels = labels[:-30, :]
test_labels = labels[-30:, :]

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3),
                        padding='SAME', input_shape=images.shape[1:], activation=tf.nn.relu),
    keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), padding='SAME', activation=tf.nn.relu),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train_images, train_labels,
          batch_size=10,
          epochs=10,
          shuffle=True)
model.evaluate(test_images, test_labels)
