import tensorflow as tf

lastDay = 1913

firstDay = 1913 - 365*2

batch_size = 256

EPOCHS = 30

data_dir = "./m5-forecasting-accuracy/"

AUTOTUNE = tf.data.experimental.AUTOTUNE
