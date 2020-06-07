import tensorflow as tf
import numpy as np
import os

from constants import *
from train_utils import *
from preprocess import Preprocessing
from datapipe import data_loader


def model(
        num_encoder_tokens=32,
        latent_dim=8,
        latent_dim2=8,
        latent_dim3=8,
        num_dense_input=7):

    # Encoder definition
    encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))

    e_outputs, h1, c1 = tf.keras.layers.LSTM(latent_dim, return_state=True,
                                             return_sequences=True, name="e1")(encoder_inputs)
    output, h2, c2 = tf.keras.layers.LSTM(
        latent_dim, return_state=True, name="e2")(e_outputs)

    # final dense layers
    decoder_dense = tf.keras.layers.Dense(latent_dim,
                                          activation='tanh', name="dense")

    decoder_outputs = decoder_dense(output)

    dense_inputs = tf.keras.Input(shape=(None, num_dense_input))

    d1 = tf.keras.layers.Dense(latent_dim2,
                               activation='tanh', name="dense2")
    d2 = tf.keras.layers.Dense(latent_dim3,
                               activation='tanh', name="dense3")
    d3 = tf.keras.layers.Dense(1,
                               activation='linear', name="dense4")

    d1out = d1(decoder_outputs)
    d2out = d2(d1out)
    outputs = d3(d2out)

    # define the model
    model = tf.keras.Model([encoder_inputs, dense_inputs], outputs)

    return model


def preprpcessing():
    datas = Preprocessing()

    datas.standardize()
    datas.encode_all()

    train_ds, val_ds = data_loader(datas.sales_df, datas.calendar_df)
    return train_ds, val_ds


class Train():
    def __init__(self):
        self.num_encoder_tokens = 5
        self.model = model(num_encoder_tokens=self.num_encoder_tokens,
                           latent_dim=32, latent_dim2=32, latent_dim3=8)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="val_loss")

    @tf.function
    def train_step(self, x, x_2, t):
        with tf.GradientTape() as tape:
            y = self.model([x, x_2])
            t_loss = self.loss(t, y)
        gradients = tape.gradient(t_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        self.train_loss(t_loss)

    @tf.function
    def test_step(self, x, x_2, t):
        x_ = tf.concat(
            [tf.expand_dims(tf.zeros_like(x[:, 0]), 1), x[:, :0:-1]], 1
        )
        y = self.model([x, x_2])
        t_loss = self.loss(t, y)
        self.test_loss(t_loss)

    def train(self, train_ds, val_ds):
        checkpoint_dir = './training_checkpoint_end2end'
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         model=self.model)

        early_stopping = EarlyStopping(5)

        file_out = open("output.txt", "w")

        with tf.device('gpu:0'):
            for epoch in range(EPOCHS):
                n = 0
                for x, x_2, y in train_ds:
                    self.train_step(x, x_2, y)
                    n += 1
                    print("epoch:", epoch, "batch:", n, end="\r", flush=True)

                for x, x_2, y in val_ds:
                    self.test_step(x, x_2, y)

                print(" ")

                template = "Epoch: {}, Loss: {}, Test Loss: {}"
                print(template.format(epoch+1,
                                      self.train_loss.result(),
                                      self.test_loss.result()
                                      ), file=file_out, flush=True)

                if (epoch + 1) % 5 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)

                if early_stopping(self.test_loss.result().numpy()):
                    print("Early stopping at epoch ", epoch, flush=True)
                    break

                self.train_loss.reset_states()
                self.test_loss.reset_states()

        checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
    train_ds, val_ds = preprpcessing()
    train_lstm = Train()
    train_lstm.train(train_ds, val_ds)
