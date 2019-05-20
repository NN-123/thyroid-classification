import tensorflow as tf
from tensorflow import keras
import loader

data = loader.load()
(train_images, train_labels), (test_images, test_labels) = loader.make_dataset(data)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3),
                        padding='SAME', input_shape=train_images.shape[1:], activation=tf.nn.relu),
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

model = keras.applications.InceptionV3()
model = keras.Model()
