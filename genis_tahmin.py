import numpy as np
import tensorflow as tf

class_names = ['kültür', 'toprak', 'yabancı']
img_height = img_width = boy = 180
model = tf.keras.models.load_model("mymodel.keras")

# Kullanım:
# percent, class_name = tahmin("")
def tahmin(croppedimage):
    img_array = tf.keras.utils.img_to_array(croppedimage)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return [100*np.max(score), class_names[np.argmax(score)]]

from math import floor
import PIL
from PIL import Image, ImageDraw, ImageFont
fp = r"D:\Business\datasets\CornRawDataset\2022-08-22-114709-55.jpg"
image = Image.open(fp=fp)
drw = ImageDraw.Draw(image, "RGBA")
cols = floor(image.width / img_height)
rows = floor(image.height / img_height)

font = ImageFont.truetype("arial.ttf", 30)

for r in range(rows):
    for c in range(cols):
        x = c * boy
        y = r * boy
        alan = (x, y, x+boy, y+boy)
        cropped = image.crop(alan)
        percent, class_name = tahmin(cropped)
        if class_name == "yabancı":
            drw.rectangle(alan, (255,0,0,50))
            drw.text((x+10,y+10), "{:.2f}".format(percent), font=font)
        elif class_name == "kültür":
            drw.rectangle(alan, (0, 255, 0, 50))
            drw.text((x + 10, y + 10), "{:.2f}".format(percent), font=font)


image.show()





