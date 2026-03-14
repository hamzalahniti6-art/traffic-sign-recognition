import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Bildgröße
IMG_SIZE = 30

data = []
labels = []

train_path = "dataset/Train"

print("Lade Trainingsbilder...")

# Bilder laden
for folder in os.listdir(train_path):
    path = os.path.join(train_path, folder)
    if os.path.isdir(path):
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(int(folder))
            except:
                pass

data = np.array(data)
labels = np.array(labels)

print("Anzahl Bilder:", len(data))

# Daten normalisieren
data = data / 255.0

# Labels umwandeln
labels = to_categorical(labels, 43)

# Trainings und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Baue CNN Modell...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(43, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starte Training...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

print("Training abgeschlossen!")

loss, acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", acc)