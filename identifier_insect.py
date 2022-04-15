from random import Random
import random
from pandas  import read_csv
import numpy as np
import matplotlib.pyplot as plot
import tensorflow_hub as hub
from tensorflow.python.keras import layers
import tensorflow as tf
import PIL.Image as Image

class IdentifierInsect():
    def identifier(id,imageUrl):
        classifier_url ="https://tfhub.dev/google/aiy/vision/classifier/insects_V1/1"
        IMAGE_SHAPE = (224, 224)
        label_insects= read_csv("aiy_insects_V1_labelmap.csv")
        classifier = tf.keras.Sequential([
            hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
        ])
        idx = random.randint(0,100)
        imageInsect = tf.keras.utils.get_file(f"{id}{str(idx)}.jpg",imageUrl)
        imageInsect = Image.open(imageInsect).resize(IMAGE_SHAPE)
        imageInsect = np.array(imageInsect)/255.0
        result = classifier.predict(imageInsect[np.newaxis, ...])
        predicted_class = np.argmax(result[0], axis=-1)
        val = label_insects[label_insects['id'] == predicted_class].to_numpy()
        return val[0][1]


