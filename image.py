import os
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
import random

master_dir = "Data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train"
img_dirs = os.listdir( master_dir )

def load_data(val_split=0.1, num_classes=4):

    x_train = np.array()
    y_train = np.array()
    x_test = np.array()
    y_test = np.array()

    for img_dir in img_dirs:
        img_names = os.listdir( os.path.join( master_dir , img_dir ) )
        for name in img_names:
            img_path = os.path.join( master_dir , img_dir , name )
            image = Image.open( img_path ).resize( ( 224 , 224 ) ).convert( 'L' )
            array2d = np.array(image)
            array1d = array2d.ravel()
            np.append(x_train, array1d)
            np.append(y_train, img_dir)
            print(array1d.size)
            # Store this image in an array with its corresponding label

    for x in range(val_split * y_train.size):
        index = random.randint(0, y_train.size - 1)
        np.append(x_test, x_train[index])
        np.append(y_test, y_train[index])
        np.delete(x_train, index)
        np.delete(y_train, index)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)