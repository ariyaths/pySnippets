# Basic Py Libraries
import os
from datetime import datetime
import pandas as pd
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import seaborn as sns

# Import and Download NLTK resources
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# Basic-ML and Neural Net Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from keras import datasets, mixed_precision, Sequential, Input
from keras.models import Model, load_model
from keras.initializers import Constant, RandomNormal, RandomUniform
from keras.layers import Rescaling, Conv2D, BatchNormalization, MaxPooling2D, \
                  Dropout, GlobalAveragePooling2D, Flatten, Dense, Activation, \
                  RandomFlip, RandomTranslation, RandomRotation, RandomZoom, \
                  RandomCrop, Add, ZeroPadding2D, TextVectorization, LSTM, GRU, \
                  Embedding, LayerNormalization, Layer, GlobalAveragePooling1D, \
                  GlobalMaxPooling1D
from keras.optimizers import AdamW, schedules, Adamax
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from livelossplot import PlotLossesKerasTF

# [nltk_data] Downloading package stopwords to /home/vigy/nltk_data...
# [nltk_data]   Unzipping corpora/stopwords.zip.
# [nltk_data] Downloading package wordnet to /home/vigy/nltk_data...
# [nltk_data] Downloading package omw-1.4 to /home/vigy/nltk_data...
# [nltk_data] Downloading package averaged_perceptron_tagger to
# [nltk_data]     /home/vigy/nltk_data...
# [nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
# [nltk_data] Downloading package averaged_perceptron_tagger_eng to
# [nltk_data]     /home/vigy/nltk_data...
# [nltk_data]   Unzipping taggers/averaged_perceptron_tagger_eng.zip.
# 2025-12-11 10:45:17.590100: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
