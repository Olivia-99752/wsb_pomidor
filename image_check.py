from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Zaladowanie modelu
model_path = "pomidor_model.h5"
model = load_model(model_path)

# Nazwy klas
class_names = ['Immature', 'Mature']

# Zaladowanie oraz dopasowanie rozmiaru obrazu, normalizacja
test_image = image.load_img('tomato.jpg', target_size=(512, 512))
test_image = image.img_to_array(test_image) / 255.0

# Dodanie "dimension", aby obraz byl rozpoznawany jako batch (1, 512, 512, 3)
test_image = np.expand_dims(test_image, axis=0)

# Predykcja
result = model.predict(test_image, batch_size=1)

print(result)

# Wskazanie predykcji
predicted_class = np.argmax(result, axis=1)[0]

print(f'Predicted class: {class_names[predicted_class]}')

# Wizualizacja
img = image.array_to_img(test_image[0])
plt.imshow(img)
plt.title(f"Predicted: {class_names[predicted_class]}")
plt.show()
