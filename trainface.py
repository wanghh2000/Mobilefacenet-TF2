import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from model.mobilefacenet import ArcFace_v2
from model.mobilefacenet_func import mobilefacenet
from sklearn.model_selection import train_test_split
from test_lfw import get_features, evaluation_10_fold

#from tensorflow.keras.mixed_precision import experimental as mixed_precision

# CONFIG
# flag for wether load pre-train model or not
# LOAD_MODEL = 0
# pre-train model
PRE_MODEL = "pretrained_model/training_model/inference_model.h5"
CHKP_MODEL = "pretrained_model/saved_model/best_model_.{epoch:02d}-{val_loss:.2f}.h5"
FINAL_MODEL = 'pretrained_model/saved_model/inference_model.h5'
RESUME = False
BATCHSZIE = 16
#EPOCHS = 70
EPOCHS = 70
# load dataset
LFW_DIR = 'C:/bd_ai/ds/lfw'
CASIA = 'C:/bd_ai/ds/CASIA'
CASIADIR = os.path.join(CASIA, 'CASIA-WebFace-112X96')
CASIATXT = os.path.join(CASIA, 'CASIA-WebFace-112X96.txt')

'''
MIXED_PRECISION = False
if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
'''


def load_dataset(val_split=0.05):
    image_list = []     # image directory
    label_list = []     # label
    with open(CASIATXT) as f:
        img_label_list = f.read().splitlines()
    for info in img_label_list:
        image_dir, label_name = info.split(' ')
        img_dir = os.path.join(CASIADIR, image_dir)
        image_list.append(img_dir)
        label_list.append(int(label_name))

    trainX, testX, trainy, testy = train_test_split(image_list, label_list, test_size=val_split)

    return trainX, testX, trainy, testy


def preprocess(x, y):
    # x: directory，y：label
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)  # RGBA
    x = tf.image.resize(x, [112, 96])

    x = tf.image.random_flip_left_right(x)

    # x: [0,255]=> -1~1
    x = (tf.cast(x, tf.float32) - 127.5) / 128.0
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, cls_num, on_value=None, off_value=None)

    if RESUME:
        return (x, y), y
    else:
        return x, y


def mobilefacenet_train(softmax=False):
    # build train model
    if RESUME:
        model = load_model(PRE_MODEL)
        inputs = model.input
        x = model.output
    else:
        x = inputs = Input(shape=(112, 96, 3))
        x = mobilefacenet(x)

    if softmax:
        x = Dense(cls_num)(x)
        outputs = Activation('softmax', dtype='float32', name='predictions')(x)
        return Model(inputs, outputs, name='mobile_face_net')
    else:
        y = Input(shape=(cls_num,), name="target")
        outputs = ArcFace_v2(n_classes=cls_num)((x, y))
        return Model([inputs, y], outputs, name='mobile_face_net')


# get data slices
train_image, val_image, train_label, val_lable = load_dataset()
train_count = len(train_image)
val_count = len(val_image)
# get class number
cls_num = len(np.unique(train_label))
# construct train dataset
db_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))
db_train = db_train.shuffle(BATCHSZIE).repeat().map(preprocess).batch(BATCHSZIE)
db_val = tf.data.Dataset.from_tensor_slices((val_image, val_lable))
db_val = db_val.shuffle(BATCHSZIE).repeat().map(preprocess).batch(BATCHSZIE)
print('cls_num=%d, train_count=%d, val_count=%d, BATCHSZIE=%d' % (cls_num, train_count, val_count, BATCHSZIE))


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


'''
class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs=None):
        model.save_weights("pretrained_model/", save_format="tf")
'''


class TestLWF(tf.keras.callbacks.Callback):
    # test on LWF
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs=None):
        if RESUME:
            outs = model.layers[-3].output
            infer_model = Model(inputs=model.input[0], outputs=outs)
        else:
            outs = model.layers[-3].output
            infer_model = Model(inputs=model.input, outputs=outs)

        result_mat = 'result/best_result.mat'
        get_features(model=infer_model, lfw_dir=LFW_DIR, feature_save_dir=result_mat)
        evaluation_10_fold()


def scheduler(epoch):
    # decay scheduler
    # [36, 52, 58]
    if RESUME:
        if epoch < 36:
            return 0.1
        elif epoch < 52:
            return 0.01
        elif epoch < 58:
            return 0.001
        else:
            return 0.0001
    else:
        if epoch < 20:
            return 0.1
        elif epoch < 35:
            return 0.01
        elif epoch < 45:
            return 0.001
        else:
            return 0.0001


if __name__ == '__main__':
    if RESUME:
        model = load_model(PRE_MODEL)
    else:
        model = mobilefacenet_train(softmax=True)
    # model.summary()

    tbCallBack = TensorBoard(log_dir="logs", histogram_freq=1, write_images=True)
    history = LossHistory()
    # , save_weights_only=True),
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50)
    checkpoint = ModelCheckpoint(CHKP_MODEL, monitor='val_loss'),
    learnrate = LearningRateScheduler(scheduler)
    #reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=200, min_lr=0)
    #callback_list = [earlystop, checkpoint, learnrate, reducelr, history, TestLWF()]
    callback_list = [tbCallBack, earlystop, checkpoint, learnrate, TestLWF()]
    #callback_list = [tbCallBack, earlystop, checkpoint, learnrate]

    # compile model
    #optimizer = tf.keras.optimizers.Adam(lr=0.001, epsilon=1e-8)
    optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    LOSS_STR = 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=LOSS_STR, metrics=['accuracy'])
    #model.fit(db_train, validation_data=db_val, validation_freq=1, epochs=EPOCHS, callbacks=callback_list, initial_epoch=34)
    model.fit(db_train,
            initial_epoch=0,
            epochs=EPOCHS,
            steps_per_epoch=int(train_count/BATCHSZIE),
            validation_data=db_val,
            validation_steps=int(val_count/BATCHSZIE),
            validation_freq=1,
            callbacks=callback_list,
            workers=1,
            use_multiprocessing=False)

    # inference model save
    if RESUME:
        outs = model.layers[-3].output
        inference_model = Model(inputs=model.input[0], outputs=outs)
    else:
        outs = model.layers[-3].output
        inference_model = Model(inputs=model.input, outputs=outs)
    inference_model.save(FINAL_MODEL)
