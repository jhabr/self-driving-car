import cv2
import numpy as np
import os
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime


class ImageProcessor:

    def __init__(self):
        self.image = None

    def decode_image(self, image):
        self.image = Image.open(BytesIO(base64.b64decode(image)))

    def save_image(self):
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join('../images/drive', timestamp)
        self.image.save('{}.jpg'.format(image_filename))

    def process_image(self):
        self.to_array()
        # self.crop()
        self.resize()
        # self.rgb_to_yuv()

        return np.array([self.image])

    def to_array(self):
        self.image = np.asarray(self.image)

    def crop(self):
        self.image = self.image[60:-25, :, :]

    def resize(self):
        self.image = cv2.resize(self.image, (160, 310), cv2.INTER_AREA)

    def rgb_to_yuv(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
