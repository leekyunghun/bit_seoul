from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img("./data/dog_cat/dog.jpg", target_size=(224, 224))
img_cat = load_img("./data/dog_cat/cat.jpg", target_size=(224, 224))
img_suit = load_img("./data/dog_cat/suit.jpg", target_size=(224, 224))
img_onion = load_img("./data/dog_cat/onion.jpg", target_size=(224, 224))

# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_onion = img_to_array(img_onion)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

# RGB -> BGR  keras로 이미지를 사용할때는 RGB에서 BGR로 바꿔줘야함
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_onion = preprocess_input(arr_onion)

print(arr_dog.shape)   # (168, 299, 3)
print(arr_cat.shape)   # (448, 680, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_onion])
print(arr_input.shape)  # (2, 168, 299, 3)

model = VGG16()
pred = model.predict(arr_input)

print(pred)
print("pred.shape : ", pred.shape)  #  (2, 1000)

from tensorflow.keras.applications.vgg16 import decode_predictions

result = decode_predictions(pred)

print("==========================================")
print("result[0] : ", result[0])

print("==========================================")
print("result[1] : ", result[1])

print("==========================================")
print("result[2] : ", result[2])

print("==========================================")
print("result[3] : ", result[3])
