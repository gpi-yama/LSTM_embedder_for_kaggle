import tensorflow as tf


class EarlyStopping:
    def __init__(self, patience: int):
        self._patience = patience
        self._loss = float("inf")
        self._num_overshoot = 1

    def __call__(self, loss):
        if loss < self._loss:
            self._loss = loss
            self._num_overshoot = 0
            return False
        else:
            self._num_overshoot += 1
            if self._patience <= self._num_overshoot:
                return True
