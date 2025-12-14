import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from src import util
from src.evaluate import auc_rocs
from src.config import train_df, IMAGE_DIR, labels, MODEL_DIR

model = tf.keras.models.load_model(
    MODEL_DIR,
    compile=False
)

labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]
fig = util.compute_gradcam(model, '00005410_000.png', IMAGE_DIR, train_df, labels, labels_to_show, H=224, W=224)

os.makedirs("results/gradcam_examples", exist_ok=True)
fig.savefig("results/gradcam_examples/gradcam_example1.png", dpi=300, bbox_inches="tight")
plt.close(fig)