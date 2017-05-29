import tensorflow as tf
import numpy as np
from utils import *
import random


class VAETrainer:
  """
  Takes care of training a model with SGD.
  """
  def __init__(self, model, train_path, num_epochs=5, 
               batch_size=16, max_length=30, 
               lr=0.1, lr_decay=0.001,
               session=None):
    """Initialize the trainer with a model."""

    self.model = model
    self.train_path = train_path
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.max_length = max_length
    self.lr = lr
    self.lr_decay = lr_decay
    self.session = session

    self._build_optimizer()
    
    # this loads the data into memory so that we can easily shuffle it
    # if this takes too much memory, shuffle the data on disk
    # and use gzip_reader directly
    self.corpus = list(filter_len(smart_reader(train_path),
                                  max_length=max_length))

  def _build_optimizer(self):
    """Buid the optimizer."""
    self.lr_ph = tf.placeholder(tf.float32)

    # You can use SGD here (with lr_decay > 0.0) but you might
    # run into NaN losses, so choose the lr carefully.
    # self.optimizer = tf.train.GradientDescentOptimizer(
    #   learning_rate=self.lr_ph).minimize(self.model.loss)
    
    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.lr_ph).minimize(self.model.loss)

  def train(self):
    """Trains a model."""

    steps = 0
    
    for epoch_id in range(1, self.num_epochs + 1):
      
      # shuffle data set every epoch
      random.shuffle(self.corpus)
      epoch_loss = 0.0
      epoch_steps = 0

      for batch_id, batch in enumerate(iterate_minibatches(
            self.corpus, batch_size=self.batch_size), 1):

        # Dynamic learning rate, cf. Bottou (2012),
        # Stochastic gradient descent tricks.
        lr_t = self.lr * (1 + self.lr * self.lr_decay * steps)**-1
        
        x = prepare_batch_data(batch, self.model.vocabulary)
        
        feed_dict = { 
          self.lr_ph : lr_t,
          self.model.x : x
        }

        fetches = {
          "optimizer": self.optimizer,
          "loss": self.model.loss,
          "ce": self.model.ce,
          "kl": self.model.kl,
          "acc_correct": self.model.accuracy_correct,
          "acc_total": self.model.accuracy_total,
          "accuracy": self.model.accuracy,
          "predictions": self.model.predictions
        }

        res = self.session.run(fetches, feed_dict=feed_dict)

        epoch_loss += res["loss"]
        steps += 1
        epoch_steps += 1
        
        if batch_id % 100 == 0:
          print("Iter {} loss {} ce {} kl {} acc {:1.2f} {}/{} lr {:1.6f}".format(
              steps, res["loss"], res["ce"], res["kl"], res["accuracy"],
               int(res["acc_correct"]), int(res["acc_total"]), lr_t))

      print("Epoch {} epoch_loss {}".format(
        epoch_id, epoch_loss / float(epoch_steps)))

      # save parameters
      save_path = self.model.save(self.session, path="model.ckpt")
      print("Model saved in file: %s" % save_path)
