import tensorflow as tf
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from src import util
from src.prepare_data import get_test_and_valid_generator 
from src.config import train_df, valid_df, test_df, MODEL_DIR, IMAGE_DIR, labels, BATCH_SIZE

model = tf.keras.models.load_model(
    MODEL_DIR,
    compile=False
)

total_test_samples = len(test_df)
STEPS = math.ceil(total_test_samples / BATCH_SIZE)

valid_generator, test_generator = get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels, target_w=224,
    target_h=224)

predicted_vals = model.predict(test_generator, steps=STEPS)
auc_rocs= util.get_roc_curve(labels, predicted_vals, test_generator)

