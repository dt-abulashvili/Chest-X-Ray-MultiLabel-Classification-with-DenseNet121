import pandas as pd

labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia',
          'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
          'Pneumothorax', 'Pleural_Thickening', 'Pneumonia',
          'Fibrosis', 'Edema', 'Consolidation']

IMAGE_DIR = "data/nih/images-small"
WEIGHTS_DIR = 'models/nih/densenet.hdf5'
MODEL_DIR = "models/pre_trained_model.h5"

train_df = pd.read_csv("data/nih/train-small.csv")
valid_df = pd.read_csv("data/nih/valid-small.csv")
test_df = pd.read_csv("data/nih/test.csv")  

BATCH_SIZE = 8