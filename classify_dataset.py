import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define parameters for the dataset loader.
# Adjust batch size according to the memory volume of your GPU;
# 16 works well on most GPU
# 256 works well on NVIDIA RTX 3090 with 24 GB VRAM
batch_size = 256
img_width = 320
img_height = 240
IMG_SIZE = (img_height, img_width)
IMG_SHAPE = IMG_SIZE + (3,)



def get_default_model(num_trainable_layers):
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    # freeze the convolutional base
    if num_trainable_layers == 0:
        for layer in base_model.layers:
            layer.trainable = False
    elif num_trainable_layers > 0:
        for layer in base_model.layers[:-num_trainable_layers]:
            layer.trainable = False
        for layer in base_model.layers[-num_trainable_layers:]:
            layer.trainable = True
    else:
        # num_trainable_layers negative, set all to trainable
        for layer in base_model.layers:
            layer.trainable = True

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    # rescale pixel values
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Build the model
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = inputs
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    model.compile(optimizer='Adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

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
    # enable memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if model is None:
        model = get_default_model(num_trainable_layers)
    fit_model(name, model, train_ds, val_ds, test_ds, num_epochs, weights_dict)
