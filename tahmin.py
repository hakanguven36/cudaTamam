import os
import numpy as np
import tensorflow as tf

class_names = ['kültür', 'toprak', 'yabancı']
img_height = img_width = 180
model = tf.keras.models.load_model("mymodel.keras")

def tahmin(target_path):
    img = tf.keras.utils.load_img(target_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "%{:.2f} olasılıkla '{}' sınıfına aittir.".format(100*np.max(score), class_names[np.argmax(score)])
    )

img_path = r"D:\DTS\Completed - Yabancılar01\yabancı\9964216d-17ae-429e-8bf3-a63947512138.jpg"
tahmin(img_path)
