import tensorflow as tf
import numpy as np
from utils import *


class VAE:
  """A simple Variational Auto Encoder."""
  
  def __init__(self, batch_size=8, vocabulary=None, emb_dim=32, rnn_dim=32, z_dim=16):

    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.rnn_dim = rnn_dim
    self.z_dim = z_dim
    
    self.vocabulary = vocabulary
    self.vocabulary_size = len(vocabulary)    

    self._create_placeholders()
    self._create_weights()
    self._build_model()
    self.saver = tf.train.Saver()
    
  def _create_placeholders(self):
    """We define placeholders to feed the data to TensorFlow."""
    # "None" means the batches may have a variable batch size and length.
    self.x = tf.placeholder(tf.int64, shape=[None, None])

  def _create_weights(self):
    """Create all weights for this VAE."""

    self.mu_W = tf.get_variable(
      name="mu_W", initializer=tf.random_normal_initializer(),
      shape=[self.rnn_dim, self.z_dim])

    self.mu_b = tf.get_variable(
      name="mu_b", initializer=tf.random_normal_initializer(),
      shape=[self.z_dim])

    self.log_sig_sq_W = tf.get_variable(
      name="log_sig_sq_W", initializer=tf.random_normal_initializer(),
      shape=[self.rnn_dim, self.z_dim])

    self.log_sig_sq_b = tf.get_variable(
      name="log_sig_sq_b", initializer=tf.random_normal_initializer(),
      shape=[self.z_dim])
    
    self.y_W = tf.get_variable(
      name="y_W", initializer=tf.random_normal_initializer(),
      shape=[self.z_dim, self.rnn_dim])

    self.y_b = tf.get_variable(
      name="y_b", initializer=tf.random_normal_initializer(),
      shape=[self.rnn_dim])
    
    self.softmax_W = tf.get_variable(
      name="softmax_W", initializer=tf.random_normal_initializer(),
      shape=[self.rnn_dim, self.vocabulary_size])
    
    self.softmax_b = tf.get_variable(
      name="softmax_b", initializer=tf.random_normal_initializer(),
      shape=[self.vocabulary_size])    

  def save(self, session, path="model.ckpt"):
    """Saves the model."""
    return self.saver.save(session, path)
    
  def _build_model(self):
    """Builds the computational graph for our model."""
    
    # Some useful values from the input data
    batch_size = tf.shape(self.x)[0]
    longest_x = tf.shape(self.x)[1]

    x_mask = tf.cast(tf.sign(self.x), tf.float32)  # Shape: [B, M]
    x_len = tf.reduce_sum(tf.sign(self.x), axis=1)  # Shape: [B]

    # ###############################################
    # This is the **inference** network q_\phi(Z | x)
    #
    #  it predicts for each x a d-dimensional vector of means and a vector of (log) variances
    #  it does so from x's 1-hot encoding
    #  thus the first step is to embed x

    # Let's create a word embedding matrix. 
    # These are trainable parameters, so we use tf.Variable.
    embeddings = tf.get_variable(
      name="embeddings", initializer=tf.random_normal_initializer(),
      shape=[self.vocabulary_size, self.emb_dim])    
    
    # Now we start defining our graph.
    # This looks up the embedding vector for each word.
    # Shape: [batch_size, time_steps, embedding_size]
    embedded = tf.nn.embedding_lookup(embeddings, self.x)


    # Here we demonstrate that the inference network has access to the complete observation
    #  by using a BiRNN to represent each word in context

    # First we need to choose what RNN cell we use in our Bidirectional RNN.
    # We will choose an LSTM with default configuration.
    cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.rnn_dim, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.rnn_dim, state_is_tuple=True)

    # Now let's transform our word embeddings!
    # Function `tf.nn.bidirectional_dynamic_rnn` will return a sequence of 
    # hidden states for the inputs (the embeddings) that we provide.
    # We also need to give it the lengths of the sequences, otherwise
    # it would keep updating the hidden states for sentences that have no 
    # more inputs.
    # `Dynamic` means that the RNN will unroll for the required number of time steps
    # in our batch, which can be different per batch. 
    # We do not have to tell it how many time steps to unroll. 
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedded, 
      sequence_length=x_len, dtype=tf.float32)
    
    # Now let's combine the forward and backward states,
    # so that we have 1 representation per time step.
    # It is more common to concatenate, but summing saves us some parameters.
    h = outputs[0] + outputs[1]  # [B, M, rnn_dim]
    h_dim = tf.shape(h)[-1]  # [rnn_dim] or [2*rnn_dim] if we concatenate
    
    # For z, we need mu and log sigma^2
    h = tf.reshape(h, [-1, h_dim])  # [B * M, h_dim]

    # At this point, we have context-aware representations of each and every word
    #  we will use these representations to independently predict a vector of means
    #  and a vector of (log) variances (as below)

    z_mu = tf.matmul(h, self.mu_W) + self.mu_b  # [B * M, z_dim]
    z_log_sig_sq = tf.matmul(h, self.log_sig_sq_W) + self.log_sig_sq_b  # [B * M, h_dim]

    # We cannot marginalise z exactly, thus we make an MC estimate of the ELBO
    #  because we train over large collections and use mini-batches,
    #  a single sample per word position is usually sufficient to get
    #  an estimate which has low-enough variance (this is not a theoretical argument, just empirical)

    # First, we sample noise vectors from a standard Gaussian
    # (check the lecture notes if you are confused about this)
    epsilon = tf.random_normal(tf.shape(z_log_sig_sq), 0, 1, dtype=tf.float32)  # [B * M, h_dim]

    # Our sampled z is a **deterministic** function of the random noise (epsilon)
    # this pushes all sources of non-determinism out of the computational graph
    # which is very convenient
    z = z_mu + tf.sqrt(tf.exp(z_log_sig_sq)) * epsilon  # [B * M, h_dim]

    # ##############################################
    # This is the *generative* network
    #  it conditions on our sampled z to predict the parameters of a Categorical over the vocabulary

    # Here we employ one non-linear layer (but this is optional)
    h_dec = tf.matmul(z, self.y_W) + self.y_b  # [B * M, h_dim]
    h_dec = tf.tanh(h_dec)  # Shape: [B * M, h_dim]
    # and these are our logits (the input to a softmax)
    # tensorflow prefers to use logits to compute the cross entropy loss
    # but see that this is just a code optimisation probably motivated solely by numerical stability
    logits = tf.matmul(h_dec, self.softmax_W) + self.softmax_b  # Shape: [B * M, Vx]

    # ###############################################
    # This is the MC estimate of the (negative) ELBO (because we do minimisation here)
    #  it includes the MC estimate of the negative log likelihood
    x_flat = tf.reshape(self.x, [-1])
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=x_flat, logits=logits)
    x_mask_flat = tf.reshape(x_mask, [-1], name="x_mask_flat")
    masked_losses = tf.reshape(x_mask_flat * losses, tf.shape(self.x))
    # we sum CE of words in a sentence (timesteps) and take mean over samples
    ce_per_sentence = tf.reduce_sum(masked_losses, axis=1)
    ce = tf.reduce_mean(ce_per_sentence)
    self.ce = ce

    # and it includes the analytical KL between our approximate posterior and prior
    # (check lecture notes and/or the notebook for more details)
    # reminder: z_mu and z_log_sig_sq's shapes are # [B * M, z_dim]
    #           thus we sum along the second axis
    kl = -0.5 * tf.reduce_sum(
      1 + z_log_sig_sq - tf.square(z_mu) - tf.exp(z_log_sig_sq), 1)
    kl = tf.reshape(kl, tf.shape(self.x))  # reshape back to [B, M]
    # we sum KL of actual words in a sentence
    #  (that's why we multiply timesteps by a mask)
    #  and take mean over samples
    self.kl = tf.reduce_mean(tf.reduce_sum(kl * x_mask, axis=1), axis=0)

    # The total loss is the negative MC estimate of the ELBO
    self.loss = self.ce + self.kl

    # Let's keep some quantities related to the approximate posterior
    #  so we can access them later from outside
    #  but in shape [B, M, z_dim]
    self.z = tf.reshape(z, [batch_size, longest_x, self.z_dim])
    self.z_mu = tf.reshape(z_mu, [batch_size, longest_x, self.z_dim])

    # #########################################################
    # Prediction
    #
    # Note that while training is stochastic (we sample z by sampling epsilon
    #  and computing Z= mu(x) + epsilon * sigma(x)
    #  for predicted mu(x) and sigma(x)
    #
    # we will simplify *predictions* by making them deterministic
    #  that is, for *predictions only* we will pretend z can be represented by
    #  the predicted mean, i.e. Z=mu(x)
    #
    #  Why is this a simplification?
    #  * once we get an assignment to Z, the generative network applies
    #    a nonlinear layers before predicting the final softmax
    #    this means that
    #      E[softmax(f_\theta(z))\ != softmax(f_\theta(E[z]))
    #    where E[z] = mu(x)
    #
    #  The principled thing to do is to sample a few assignments (e.g. 100)
    #   and run the generative model on each assignment
    #   then use a probabilistic disambiguation rule (e.g. most-probable-sample,
    #   or MBR, etc.).
    #  Here instead we simply take the mean as a hopefully good approximation.


    # Note how we use the SAME MLP as before, but on the predicted mean
    pred_h_dec = tf.matmul(z_mu, self.y_W) + self.y_b
    pred_h_dec = tf.tanh(pred_h_dec)  # Shape: [B * M, h_dim]
    # and these are our logits (the input to a softmax)
    pred_logits = tf.matmul(pred_h_dec,
                            self.softmax_W) + self.softmax_b  # Shape: [B * M, Vx]
    pred_py_x = tf.nn.softmax(pred_logits)
    pred_py_x = tf.reshape(pred_py_x, [batch_size, longest_x, self.vocabulary_size])
    predictions = tf.argmax(pred_py_x, axis=2)
    self.predictions = predictions

    acc = tf.equal(predictions, self.x)
    self.acc_debug = acc
    acc = tf.cast(acc, tf.float32) * x_mask
    self.accuracy_correct = tf.reduce_sum(acc)
    self.accuracy_total = tf.reduce_sum(x_mask)
    self.accuracy = self.accuracy_correct / self.accuracy_total
