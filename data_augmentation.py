import os
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling
import matplotlib.pyplot as plt

# Sciezki do folderow
data_dir = "data/Original Dataset"
augmented_dir = "data/augmented3"
img_size = (512, 512)

# Tworzenie struktury folderu augmented
os.makedirs(augmented_dir, exist_ok=True)
for class_name in os.listdir(data_dir):
    class_path = os.path.join(augmented_dir, class_name)
    os.makedirs(class_path, exist_ok=True)

# Preprocessing danych, normalizacja do [0, 1]
normalization_layer = Rescaling(1.0 / 255)

# Zastosowanie losowych transformacji, obrót obrazu, rotacja, zoom
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# Augumentacja, preprocessing danych oraz zapisywanie w nowym folderze
def preprocess_and_augment(input, output, num_augmented=5):
    # Iteracja po każdej klasie obrazów
    for class_name in os.listdir(input):
        # Sciezka do klasy wejsciowej
        class_input_path = os.path.join(input, class_name)
        # Sciezka do klasy wyjsciowej
        class_output_path = os.path.join(output, class_name)

        if not os.path.isdir(class_input_path):
            continue

        # Iteracja po plikach w folderze klasy
        for file_name in os.listdir(class_input_path):
            file_path = os.path.join(class_input_path, file_name)
            # Ładowanie obrazu i zmiana jego rozmiaru do określonego wymiaru
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=img_size)
            # Konwersja obrazu do tablicy NumPy
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            # Dodanie batch dimension
            img_array = tf.expand_dims(img_array, axis=0)
            # Normalizacja obrazu przy użyciu warstwy normalizacyjnej
            img_array = normalization_layer(img_array)

            # Tworzenie sciezki obrazu oraz zapisywanie w folderze wyjsciowym
            preprocessed_file_path = os.path.join(class_output_path, f"pre_{file_name}")
            tf.keras.preprocessing.image.save_img(preprocessed_file_path, img_array[0].numpy())

            # Augmentacja obrazow oraz zapisywanie w folderze augmented
            for i in range(num_augmented):
                augmented_img = data_augmentation(img_array, training=True)
                aug_file_name = f"aug{i}_{file_name}"
                augmented_file_path = os.path.join(class_output_path, aug_file_name)
                tf.keras.preprocessing.image.save_img(augmented_file_path, augmented_img[0].numpy())

# Preprocessing i augmentacja obrazow
preprocess_and_augment(data_dir, augmented_dir, num_augmented=5)

# Przykładowa wizualizacja obrazow
def visualize_augmented_images(input, class_name):
    class_path = os.path.join(input, class_name)
    file_name = next(iter(os.listdir(class_path)))
    file_path = os.path.join(class_path, file_name)

    img = tf.keras.preprocessing.image.load_img(file_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    img_array_normalized = normalization_layer(img_array)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(img_array[0].numpy().astype("uint8"))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.imshow((img_array_normalized[0].numpy() * 255).astype("uint8"))
    plt.title("Preprocessed")
    plt.axis("off")

    for i in range(3):
        augmented_img = data_augmentation(img_array, training=True)
        plt.subplot(1, 6, i + 3)
        plt.imshow(augmented_img[0].numpy().astype("uint8"))
        plt.title(f"Augmented {i}")
        plt.axis("off")

    plt.show()

visualize_augmented_images(data_dir, "Immature")
