import os
# Set log level before import, 0-debug(default) 1-info 2-warnning 3-error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model.mobilefacenet import *
from model.mobilefacenet_func import *
from sklearn.model_selection import train_test_split

# Set CPU as available physical device
#my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
#tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
# set GPU
#my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
#tf.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')

# To find out which devices your operations and tensors are assigned to
#tf.debugging.set_log_device_placement(True)

cls_num = 10572

def mobilefacenet_train(resume=False):
    x = inputs = tf.keras.layers.Input(shape=(112, 96, 3))
    x = mobilefacenet(x)

    if not resume:
        x = tf.keras.layers.Dense(cls_num)(x)
        outputs = tf.nn.softmax(x)
        return tf.keras.models.Model(inputs, outputs)
    else:
        y = tf.keras.layers.Input(shape=(cls_num,))
        outputs = ArcFace(n_classes=cls_num)((x, y))

        return tf.keras.models.Model([inputs, y], outputs)


def preprocess(img):
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


# load image
img_yzy = preprocess(plt.imread("image/yzy.jpg"))
img_lm = preprocess(plt.imread("image/lm.jpg"))
img_steve = preprocess(plt.imread("image/yzy_2.jpg"))

img_test = preprocess(plt.imread("image/lm_2.jpg"))


if __name__ == '__main__':

    # feed forward
    #model = keras.models.load_model("pretrained_model/training_model/inference_model_0.993.h5")
    model = keras.models.load_model("pretrained_model/training_model/inference_model_0.993.h5", compile=False)
    #model.summary()
    embedding_yzy = model.predict(img_yzy)
    embedding_lm = model.predict(img_lm)
    embedding_steve = model.predict(img_steve)

    embedding_test = model.predict(img_test)
    #print(embedding_test)

    # test result
    embedding_yzy = embedding_yzy / np.expand_dims(np.sqrt(np.sum(np.power(embedding_yzy, 2), 1)), 1)
    embedding_lm = embedding_lm / np.expand_dims(np.sqrt(np.sum(np.power(embedding_lm, 2), 1)), 1)
    embedding_steve = embedding_steve / np.expand_dims(np.sqrt(np.sum(np.power(embedding_steve, 2), 1)), 1)
    embedding_test = embedding_test / np.expand_dims(np.sqrt(np.sum(np.power(embedding_test, 2), 1)), 1)

    # get result
    print(np.sum(np.multiply(embedding_yzy, embedding_test), 1))
    print(np.sum(np.multiply(embedding_lm, embedding_test), 1))
    print(np.sum(np.multiply(embedding_steve, embedding_test), 1))

    # save database
    db = np.concatenate((embedding_yzy, embedding_lm, embedding_steve), axis=0)
    print(db.shape)
    #np.save("pretrained_model/db", db)
