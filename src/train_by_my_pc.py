import sys
from pathlib import Path
import tensorflow as tf
import pandas as pd

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from src.prepare_data import get_train_generator, get_test_and_valid_generator
from src.config import labels, WEIGHTS_DIR, train_df, valid_df, test_df, IMAGE_DIR, BATCH_SIZE

base_model = DenseNet121(weights=WEIGHTS_DIR, include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(labels), activation='sigmoid')(x)

#pos_weights, neg_weights = compute_class_weights(labels)
#model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

model = Model(inputs=base_model.input, outputs=predictions) 
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())
  
train_generator = get_train_generator(
    df=train_df,
    image_dir=IMAGE_DIR,
    x_col="Image",
    y_cols=labels,
    batch_size=BATCH_SIZE,
    target_w=224,
    target_h=224
)

valid_generator, _ = get_test_and_valid_generator(
    valid_df=valid_df,
    test_df=test_df,
    train_df=train_df,
    image_dir=IMAGE_DIR,
    x_col="Image",
    y_cols=labels,
    batch_size=BATCH_SIZE,
    target_w=224,
    target_h=224
)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch=50,
    validation_steps=10,
    epochs=2
)

model.save("models/my_pc_trained_model.h5")