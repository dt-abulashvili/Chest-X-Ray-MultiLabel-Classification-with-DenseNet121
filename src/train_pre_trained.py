from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from src.prepare_data import labels

base_model = DenseNet121(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3))

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights("models/nih/pretrained_model.h5")

model.save("models/pre_trained_model.h5")