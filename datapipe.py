import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from constants import *


def data_loader(sales_df):
    def window(x):
        return np.expand_dims(timesales[x[0], x[1]:batch_size+x[1]], axis=-1)

    timesales = sales_df[[
        "d_"+str(i) for i in range(firstDay, lastDay)]].values.astype("float32")

    index = np.zeros([len(timesales), len(timesales[0]) - batch_size - 1],
                     dtype="int32")
    index = np.array(np.where(index == 0), dtype="int32").T
    index = shuffle(index, random_state=0)

    train_idx, val_idx = train_test_split(index, test_size=0.2)

    # ------- tensorflow data pipe settings ------
    train_ds = tf.data.Dataset.from_tensor_slices(train_idx)
    train_ds = train_ds.shuffle(buffer_size=1000000)
    train_ds = train_ds.map(lambda x: tf.py_function(window, [x], tf.float32))
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_idx)
    val_ds = val_ds.map(lambda x: tf.py_function(window, [x], tf.float32))
    val_ds = val_ds.batch(512).prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
