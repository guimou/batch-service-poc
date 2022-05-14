### Imports

import os
import io

import boto3
import botocore
import numpy as np
import pandas as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

### Variables

# Source Info
source_endpoint = os.getenv('SOURCE_ENDPOINT')
source_aws_access_key_id = os.getenv('SOURCE_ACCESS_KEY')
source_aws_secret_access_key = os.getenv('SOURCE_SECRET_KEY')
source_bucket = os.getenv('SOURCE_BUCKET')

source_train_data_dir = os.getenv('SOURCE_TRAIN_DATA_DIR', 'train')
source_validation_data_dir = os.getenv('SOURCE_VALIDATION_DATA_DIR', 'val')
source_test_data_dir = os.getenv('SOURCE_TEST_DATA_DIR', 'test')

# Destination Info
destination_endpoint = os.getenv('DESTINATION_ENDPOINT')
destination_aws_access_key_id = os.getenv('DESTINATION_ACCESS_KEY')
destination_aws_secret_access_key = os.getenv('DESTINATION_SECRET_KEY')
destination_bucket = os.getenv('DESTINATION_BUCKET')

destination_model_dir = os.getenv('DESTINATION_MODEL_DIR', 'model')
destination_model_name = os.getenv('DESTINATION_MODEL_DIR', 'pneumonia_model.h5')

# Temp data directory
tmp_data_dir = os.getenv('TMP_DATA_DIR', '/tmp/chest_xray')

# HyperParameters
epochs = os.getenv('EPOCHS', 20)
batch_size = os.getenv('BATCH_SIZE', 16)

### S3 Connections

s3_source = boto3.client('s3','us-east-1',
                endpoint_url = source_endpoint,
                aws_access_key_id = source_aws_access_key_id,
                aws_secret_access_key = source_aws_secret_access_key,
                use_ssl=True if 'https' in source_endpoint else False)

s3_destination = boto3.client('s3','us-east-1',
                endpoint_url = destination_endpoint,
                aws_access_key_id = destination_aws_access_key_id,
                aws_secret_access_key = destination_aws_secret_access_key,
                use_ssl=True if 'https' in destination_endpoint else False)

### Initialize variables

# dimensions of our images.
img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

### Import images

print('Importing images, please wait...')

paginator = s3_source.get_paginator('list_objects_v2')

# Train objects copy and count
train_objects_pages = paginator.paginate(Bucket=source_bucket, Prefix=source_train_data_dir)

nb_train_samples = 0

for train_objects_page in train_objects_pages:
    nb_train_samples += sum(1 for _ in train_objects_page['Contents'])
    for file in train_objects_page.get('Contents', []):
        filename = file.get('Key')
        dest_pathname = os.path.join(tmp_data_dir, filename)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        s3_source.download_file(source_bucket, file.get('Key'), dest_pathname)

print('Train samples: ' + str(nb_train_samples))

# Validation objects copy and count

validation_objects_pages = paginator.paginate(Bucket=source_bucket, Prefix=source_validation_data_dir)

nb_validation_samples = 0

for validation_objects_page in validation_objects_pages:
    nb_validation_samples += sum(1 for _ in validation_objects_page['Contents'])
    for file in validation_objects_page.get('Contents', []):
        filename = file.get('Key')
        dest_pathname = os.path.join(tmp_data_dir, filename)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        s3_source.download_file(source_bucket, file.get('Key'), dest_pathname)

print('Validation samples: ' + str(nb_validation_samples))

# Test objects copy and count

test_objects_pages = paginator.paginate(Bucket=source_bucket, Prefix=source_test_data_dir)

nb_test_samples = 0

for test_objects_page in test_objects_pages:
    nb_test_samples += sum(1 for _ in test_objects_page['Contents'])
    for file in test_objects_page.get('Contents', []):
        filename = file.get('Key')
        dest_pathname = os.path.join(tmp_data_dir, filename)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        s3_source.download_file(source_bucket, file.get('Key'), dest_pathname)

print('Test samples: ' + str(nb_test_samples))

train_data_dir = os.path.join(tmp_data_dir, source_train_data_dir)
validation_data_dir = os.path.join(tmp_data_dir, source_validation_data_dir)
test_data_dir = os.path.join(tmp_data_dir, source_test_data_dir)

### Create Model

print('Creating model...')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.layers
model.input
model.output

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

### Evaluate the model

print('Model evaluation:')
scores = model.evaluate(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

### Save the model temporarily

destination_model_filepath = os.path.join(tmp_data_dir, destination_model_dir, destination_model_name)
if not os.path.exists(destination_model_filepath):
    os.makedirs(os.path.dirname(destination_model_filepath))
model.save(destination_model_filepath)

### Upload model

model_key = destination_model_dir + '/' + destination_model_name
s3_destination.upload_file(destination_model_filepath,destination_bucket,model_key)

print('Model trained and uploaded!')