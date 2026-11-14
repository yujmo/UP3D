import os

from PIL import Image
import numpy as np
import cv2

def save_depth_as_uint16png(img, filename):
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)

def save_depth_as_uint16png_upload(img, filename):
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)