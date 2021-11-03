import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define parameters for the dataset loader.
# Adjust batch size according to the memory volume of your GPU;
# 16 works well on most GPU
# 256 works well on NVIDIA RTX 3090 with 24 GB VRAM
batch_size = 256
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
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model


def get_merged_model(num_trainable_layers=0):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    freeze_model(base_model, num_trainable_layers)

    # Build the model
    rgb_network_input = tf.keras.Input(shape=IMG_SHAPE)
    rgb_network = data_augmentation(rgb_network_input)
    rgb_network = preprocess_input(rgb_network)
    rgb_network = base_model(rgb_network, training=False)
    rgb_network = tf.keras.layers.Flatten()(rgb_network)
    rgb_network = tf.keras.Model(rgb_network_input, rgb_network)

    for layer in rgb_network.layers:
        layer._name = "rgb_" + layer.name

    of_network_input = tf.keras.Input(shape=IMG_SHAPE)
    of_network = data_augmentation(of_network_input)
    of_network = preprocess_input(of_network)
    of_network = base_model(of_network, training=False)
    of_network = tf.keras.layers.Flatten()(of_network)
    of_network = tf.keras.Model(of_network_input, of_network)

    for layer in of_network.layers:
        layer._name = "of_" + layer.name

    merged = tf.keras.layers.concatenate([rgb_network.output, of_network.output], axis=1)
    merged = tf.keras.layers.Flatten()(merged)
    #merged = tf.keras.layers.Dense(64, activation='relu')(merged)
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
    with open("results-on-test-ds-{}.txt".format(name), "w") as f:
        f.write(result_str)


def evaluate(name, train_ds, val_ds, test_ds, weights_dict={}, num_epochs=10, num_trainable_layers=0, model=None):
    if model is None:
        model = get_default_model(num_trainable_layers)

    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    fit_model(name, model, train_ds, val_ds, test_ds, num_epochs, weights_dict)
