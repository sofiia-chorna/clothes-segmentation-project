import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from PIL import Image

class Segmentator(object):
    def __init__(self, image):
        self.image = Image.open(image)
        self.model = load_model("model/save_ckp_frozen.h5")

    # limited to top wear and full body dresses (wild and studio working)
    def get_dress(self):
        tf_image = np.array(self.image)
        file = tf.image.resize_with_pad(tf_image, target_height=512, target_width=512)
        rgb = file.numpy()
        file = np.expand_dims(file, axis=0) / 255.
        seq = self.model.predict(file)
        seq = seq[3][0, :, :, 0]
        seq = np.expand_dims(seq, axis=-1)
        c1x = rgb * seq
        c2x = rgb * (1 - seq)
        cfx = c1x + c2x
        rgbs = np.concatenate((cfx, seq * 255.), axis=-1)
        result_file = self.save_image(rgbs)
        return result_file

    def save_image(self, rgbs, filename='out.png'):
        # save to file
        cv2.imwrite(filename, rgbs)

        # return file with result
        file = Image.open(filename)
        return file
