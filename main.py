import tensorflow as tf
#from tensorflow import keras
from keras import layers, models, Input
import matplotlib.pyplot as plt

# Ustawienia podstawowe
batch_size = 32
img_height = 512
img_width = 512
data_dir = 'data/augmented3'  # katalog z danymi

# Podział na zbiór treningowy i walidacyjny/testowy
# parametry 'validation_split' oraz 'subset' umożliwiają automatyczny podział
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 20% danych na walidację/test
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Informacje o klasach
class_names = train_ds.class_names
print("Klasy:", class_names)

# Optymalizacja pracy z danymi
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalizacja pikseli
normalization_layer = layers.Rescaling(1./255)

# Budowa prostego modelu CNN
model = models.Sequential([
    Input(shape=(img_height, img_width, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Trenowanie modelu
epochs = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Wyświetlanie wyników
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(range(epochs), acc, label='Dokładność treningu')
plt.plot(range(epochs), val_acc, label='Dokładność walidacji')
plt.legend(loc='lower right')
plt.title('Dokładność treningu i walidacji')

plt.subplot(2, 1, 2)
plt.plot(range(epochs), loss, label='Strata treningu')
plt.plot(range(epochs), val_loss, label='Strata walidacji')
plt.legend(loc='upper right')
plt.title('Strata treningu i walidacji')
plt.xlabel('Epoka')
plt.show()

# Ewaluacja modelu na zbiorze walidacyjnym (jeżeli walidacyjny traktujemy jako testowy)
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print("\nDokładność na zbiorze testowym:", test_acc)

# Zapisanie modelu
model.save("pomidor_model.h5")
