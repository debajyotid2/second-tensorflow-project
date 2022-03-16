from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# code to handle tensorflow GPU errors

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# end code to handle tensorflow GPU errors


# load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# preprocess dataset
X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #  brightness_range=(0.3, 0.8),
    #  shear_range=40,
    zoom_range=0.5,
    #  channel_shift_range=90,
    horizontal_flip=True,
    #  vertical_flip=True,
    validation_split=0.2
)
datagen.fit(X_train)

# one-hot encode y labels for use with categorical crossentropy

one_hot_encoder = OneHotEncoder(sparse=False)
y_train_encoded = one_hot_encoder.fit_transform(y_train)

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='tanh'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='tanh'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# train model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model
history_callback = model.fit(datagen.flow(X_train, y_train_encoded, batch_size=128, subset='training'),
                             validation_data=datagen.flow(X_train, y_train_encoded, batch_size=8, subset='validation'), epochs=100)
loss_history, accuracy_history = history_callback.history[
    "loss"], history_callback.history["accuracy"]

# test model
preds = np.argmax(model.predict(X_test), axis=1)[:, np.newaxis]
print(f"Accuracy = {np.sum(y_test == preds)/len(y_test)*100 :.2f} %")

# plot loss and accuracy
loss_history, accuracy_history = history_callback.history[
    "loss"], history_callback.history["accuracy"]
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(list(range(len(loss_history))), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(list(range(len(accuracy_history))), accuracy_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("loss_accuracy.png", bbox_inches="tight", dpi=200)

# save trained model
model.save("cifar10_cnn.h5")
