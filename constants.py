import tensorflow as tf

lastDay = 1913

firstDay = 1913 - 365*2

batch_size = 32

EPOCHS = 10

data_dir = "./m5-forecasting-accuracy/"

AUTOTUNE = tf.data.experimental.AUTOTUNE
