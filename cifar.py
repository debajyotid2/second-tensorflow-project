import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# preprocess dataset
X_train, X_test = X_train[:, :, :, 0], X_test[:, :, :, 0]
X_train, X_test = X_train.astype(
    'float32')/255.0, X_test.astype('float32')/255.0

# train, val, test split
val_size = len(y_test)
val_idx = np.random.randint(0, len(y_train), size=len(y_train))
X_val, y_val = X_train[val_idx[:val_size]], y_train[val_idx[:val_size]]
X_train, y_train = X_train[val_idx[val_size:]], y_train[val_idx[val_size:]]


# build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
history_callback = model.fit(X_train, y_train, epochs=50)
loss_history, accuracy_history = history_callback.history[
    "loss"], history_callback.history["accuracy"]

# validate model
val_error, val_acc = model.evaluate(X_val, y_val)
print(f"Valication accuracy = {val_acc*100:.3f} %")

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
model.save("cifar10.h5")
