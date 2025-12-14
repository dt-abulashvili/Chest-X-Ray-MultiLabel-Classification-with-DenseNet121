import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def check_for_leakage(df1: pd.DataFrame, df2: pd.DataFrame, patient_col: str) -> bool:
    df1_patients_unique = set(df1[patient_col])
    df2_patients_unique = set(df2[patient_col])

    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    if patients_in_both_groups:
        leakage = True
    else:
        leakage = False

    return leakage

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w=320, target_h=320):
    print("getting train generator...") 
    image_generator = ImageDataGenerator(samplewise_center=True,
                                         samplewise_std_normalization=True)
    
    generator = image_generator.flow_from_dataframe(dataframe=df,
                                                    directory=image_dir,
                                                    x_col=x_col,
                                                    y_col=y_cols,
                                                    class_mode='raw',
                                                    target_size=(target_w, target_h),
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    seed=seed)
    return generator

def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w=320, target_h=320):
    print("getting valid and test generators...") 
    raw_image_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=image_dir,
        x_col="Image",
        y_col=y_cols,
        class_mode='raw', 
        target_size=(target_w, target_h), 
        batch_size=sample_size, 
        shuffle=True)
    
    batch = next(raw_image_generator)
    sample_images = batch[0]

    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    
    image_generator.fit(sample_images)

    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode='raw',
        target_size=(target_w, target_h),
        batch_size=batch_size,
        shuffle=False,
        seed=seed)
    
    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode='raw',
        target_size=(target_w, target_h),
        batch_size=batch_size,
        shuffle=False,
        seed=seed)
    
    return valid_generator, test_generator

def compute_class_freqs(labels):
    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies

def compute_class_weights(labels):
    pos_freqs, neg_freqs = compute_class_freqs(labels)
    pos_weights = neg_freqs
    neg_weights = pos_freqs

    return pos_weights, neg_weights

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += K.mean(-(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) 
                             + neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)))
            
        return loss
    return weighted_loss
