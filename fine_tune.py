from tensorflow import keras
import loader

data = loader.load('inception')
(train_x, train_y), (test_x, test_y) = loader.make_dataset(data)

base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
dense = keras.layers.GlobalAveragePooling2D()(base_model.output)
dense = keras.layers.Dense(128, activation='relu')(dense)
predictions = keras.layers.Dense(2, activation='softmax')(dense)
model = keras.models.Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
    train_x, train_y, batch_size=15, epochs=3, shuffle=True
)
model.evaluate(test_x, test_y)

for layer in model.layers[:250]:
    layer.trainable = False
for layer in model.layers[250:]:
    layer.trainable = True

sgd = keras.optimizers.SGD(lr=0.00001, momentum=0.9)
model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    train_x, train_y, batch_size=15, epochs=10, shuffle=True
)
model.evaluate(test_x, test_y)
