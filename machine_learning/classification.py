import tensorflow as tf
import os
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models

AUTOTUNE = tf.data.AUTOTUNE


class ImageClassification:
    def __init__(self, path_images, path_annotations, class_names, img_size, batch_size = 32):
        self.path_images = path_images
        self.path_annotations = path_annotations
        self.class_names = class_names
        self.batch_size = batch_size
        self.img_size = img_size

        self.images_ds = tf.data.Dataset.list_files('RoadSignsPascalVOC/images/*/*', shuffle=900)
        self.train_size = int(len(self.images_ds)*0.8+3)

    def prepare_dataset(self, augment=False):
        train_ds = self.images_ds.take(self.train_size)
        test_ds = self.images_ds.skip(self.train_size)
        train_ds = train_ds.map(self._process_path, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(self._process_path, num_parallel_calls=AUTOTUNE)
        

        return train_ds, test_ds

    def _process_path(self,file_path):
        label = self._get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self._decode_img(img)
        return img, label

    def _get_label(self, file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def _decode_img(self,img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [280, 280])

    def optimize_datasets(self, train_ds, test_ds, augment=False):
        train_ds = self._configure_for_performance(train_ds, augment)
        test_ds = self._configure_for_performance(test_ds, augment)

        return train_ds, test_ds
        
    def _configure_for_performance(self, ds, augment=False):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)

        if augment:
            ds = ds.map(self._augment)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def _augment(self, image, label):
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
        image = tfa.image.rotate(image, tf.constant(np.pi/16))

        return image, label

    

    def create_base_model(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(280, 280, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

        return model

    def model_compile(self, model, train_ds, test_ds,epochs=10):
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model.fit(train_ds, epochs=epochs, 
                            validation_data=(test_ds))

    def create_model(self):
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal",
                            input_shape=(280,
                                        280,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(
            0.2, 0.3, fill_mode='reflect',
            interpolation='bilinear'),
            layers.experimental.preprocessing.RandomContrast(factor=0.1,),

        ])

        model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2)
        ])

        return model