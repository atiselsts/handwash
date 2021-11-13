import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import Layer

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define parameters for the dataset loader.
# Adjust batch size according to the memory volume of your GPU;
# 16 works well on most GPU
# 256 works well on NVIDIA RTX 3090 with 24 GB VRAM
batch_size = 32
img_width = 320
img_height = 240
IMG_SIZE = (img_height, img_width)
N_CHANNELS = 3
IMG_SHAPE = IMG_SIZE + (N_CHANNELS,)

N_CLASSES = 7

# data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# rescale pixel values
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def freeze_model(model, num_trainable_layers):
    if num_trainable_layers == 0:
        for layer in model.layers:
            layer.trainable = False
    elif num_trainable_layers > 0:
        for layer in model.layers[:-num_trainable_layers]:
            layer.trainable = False
        for layer in model.layers[-num_trainable_layers:]:
            layer.trainable = True
    else:
        # num_trainable_layers negative, set all to trainable
        for layer in model.layers:
            layer.trainable = True



def get_default_model(num_trainable_layers=0):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    freeze_model(base_model, num_trainable_layers)

    # Build the model
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = inputs
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l1_l2')(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model


def get_time_distributed_model1(num_frames, num_trainable_layers=0):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   pooling='avg',
                                                   weights='imagenet')

    freeze_model(base_model, num_trainable_layers)

    # Build the model
    INPUT_SHAPE = (num_frames,) + IMG_SHAPE
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = inputs
    # TODO: how to do data augmentation here?!
    # x = data_augmentation(x)
    x = preprocess_input(x)
    x = tf.keras.layers.TimeDistributed(base_model)(x)
    x = tf.keras.layers.GRU(256)(x)
#    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l1_l2')(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model

class MobileNetPreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MobileNetPreprocessingLayer, self).build(input_shape)

    def call(self, x):
        return (x / 127.5) - 1.0

    def compute_output_shape(self, input_shape):
        return input_shape


def get_time_distributed_model(num_frames, num_trainable_layers=0):
    mnet_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   pooling='avg',
                                                   weights='imagenet')

    freeze_model(mnet_model, num_trainable_layers)


    base_inputs = tf.keras.Input(IMG_SHAPE)
    x = base_inputs
    x = data_augmentation(x)
    x = MobileNetPreprocessingLayer()(x)
    base_outputs = mnet_model(x, training=False)
    base_model = tf.keras.Model(base_inputs, base_outputs)


    # Build the model
    INPUT_SHAPE = (num_frames,) + IMG_SHAPE
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = inputs
    x = tf.keras.layers.TimeDistributed(base_model)(x)
    x = tf.keras.layers.GRU(256)(x)
#    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l1_l2')(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model



def get_merged_model(num_trainable_layers=0):
    rgb_base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

    of_base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')

    freeze_model(rgb_base_model, num_trainable_layers)
    freeze_model(of_base_model, num_trainable_layers)

    # Build the model
    rgb_network_input = tf.keras.Input(shape=IMG_SHAPE)
    rgb_network = data_augmentation(rgb_network_input)
    rgb_network = preprocess_input(rgb_network)
    rgb_network = rgb_base_model(rgb_network, training=False)
    rgb_network = tf.keras.layers.Flatten()(rgb_network)
    rgb_network = tf.keras.Model(rgb_network_input, rgb_network)

    for layer in rgb_network.layers:
        layer._name = "rgb_" + layer.name

    of_network_input = tf.keras.Input(shape=IMG_SHAPE)
    of_network = data_augmentation(of_network_input)
    of_network = preprocess_input(of_network)
    of_network = of_base_model(of_network, training=False)
    of_network = tf.keras.layers.Flatten()(of_network)
    of_network = tf.keras.Model(of_network_input, of_network)

    for layer in of_network.layers:
        layer._name = "of_" + layer.name

    merged = tf.keras.layers.concatenate([rgb_network.output, of_network.output], axis=1)
    merged = tf.keras.layers.Flatten()(merged)
#    merged = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l1_l2')(merged)
    merged = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(merged)

    model = tf.keras.Model([rgb_network.input, of_network.input], merged)
    print(model.summary())

    return model


def fit_model(name, model, train_ds, val_ds, test_ds, num_epochs, weights_dict):
    # callbacks to implement early stopping and saving the model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(monitor='val_accuracy', mode='max',
                         verbose=1, save_freq='epoch',
                         filepath=name+'.{epoch:02d}-{val_accuracy:.2f}.h5')

    print("fitting the model...")
    history = model.fit(train_ds,
                        epochs=num_epochs,
                        validation_data=val_ds,
                        class_weight=weights_dict,
                        callbacks=[es, mc])

    # visualise accuracy
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(8, 8))
    plt.grid(True, axis="y")
    plt.subplot(2, 1, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.savefig("accuracy-{}.pdf".format(name), format="pdf")


    test_loss, test_accuracy = model.evaluate(test_ds)
    result_str = 'Test loss: {} accuracy: {}\n'.format(test_loss, test_accuracy)
    print(result_str)
    with open("results-{}.txt".format(name), "w") as f:
        f.write(result_str)

    measure_performance("validation", name, model, val_ds)
    measure_performance("test", name, model, test_ds)


def measure_performance(ds_name, name, model, ds, num_classes=N_CLASSES):
#    n = 0
    matrix = [[0] * num_classes for i in range(num_classes)]

#    predicted = model.predict(ds)

#    y_true = []
#    for batch in ds:
#      _, labels = batch
#      for label in labels:
#        y = int(np.argmax(label))
#        y_true.append(y)

#    y_pred = []
#    for p in predicted:
#       y_pred.append(int(np.argmax(p)))

    y_predicted = []
    y_true = []

    n = 0
    for batch in ds:
        b1, b2 = batch
        predicted = model.predict(b1)
        for y_p, y_t in zip(predicted, b2):
            y_predicted.append(int(np.argmax(y_p)))
            y_true.append(int(np.argmax(y_t)))
            n += 1

    for y_p, y_t in zip(y_predicted, y_true):
        matrix[y_t][y_p] += 1

    print("Confusion matrix:")
    for row in matrix:
        print(row)

    f1_scores = []
    for i in range(num_classes):
        total = sum(matrix[i])
        true_predictions = matrix[i][i]
        total_predictions = sum([matrix[j][i] for j in range(num_classes)])
        if total:
            precision = true_predictions / total
        else:
            precision = 0
        if total_predictions:
            recall = true_predictions / total_predictions
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        print("{} precision={:.2f}% recall={:.2f}% f1={:.2f}".format(i, 100 * precision, 100 * recall, f1))
        f1_scores.append(f1)
    s = "Average {} F1 score: {:.2f}\n".format(ds_name, np.mean(f1_scores))
    print(s)
    with open("results-{}.txt".format(name), "a+") as f:
       f.write(s)


def evaluate(name, train_ds, val_ds, test_ds, weights_dict={}, num_epochs=20, num_trainable_layers=0, model=None):
    if model is None:
        model = get_default_model(num_trainable_layers)

    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    fit_model(name, model, train_ds, val_ds, test_ds, num_epochs, weights_dict)
