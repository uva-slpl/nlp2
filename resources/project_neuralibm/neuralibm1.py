import numpy as np
import tensorflow as tf
from aer import read_naacl_alignments, AERSufficientStatistics
from utils import iterate_minibatches, prepare_data


class NeuralIBM1Model:
  """Our Neural IBM1 model."""
  
  def __init__(self, batch_size=8, 
               x_vocabulary=None, y_vocabulary=None, 
               emb_dim=32, mlp_dim=64,
               session=None):

    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.mlp_dim = mlp_dim
    
    self.x_vocabulary = x_vocabulary
    self.y_vocabulary = y_vocabulary
    self.x_vocabulary_size = len(x_vocabulary)
    self.y_vocabulary_size = len(y_vocabulary)

    self._create_placeholders()
    self._create_weights()
    self._build_model()
    
    self.saver = tf.train.Saver()
    self.session = session
    
  def _create_placeholders(self):
    """We define placeholders to feed the data to TensorFlow."""
    # "None" means the batches may have a variable maximum length.
    self.x = tf.placeholder(tf.int64, shape=[None, None])
    self.y = tf.placeholder(tf.int64, shape=[None, None])
    
  def _create_weights(self):
    """Create weights for the model."""
    with tf.name_scope("MLP") as scope:
      self.mlp_W_ = tf.get_variable(
        name="W_", initializer=tf.random_normal_initializer(),
        shape=[self.emb_dim, self.mlp_dim])

      self.mlp_b_ = tf.get_variable(
        name="b_", initializer=tf.random_normal_initializer(),
        shape=[self.mlp_dim])
      
      self.mlp_W = tf.get_variable(
        name="W", initializer=tf.random_normal_initializer(),
        shape=[self.mlp_dim, self.y_vocabulary_size])

      self.mlp_b = tf.get_variable(
        name="b", initializer=tf.random_normal_initializer(),
        shape=[self.y_vocabulary_size])    

  def save(self, session, path="model.ckpt"):
    """Saves the model."""
    return self.saver.save(session, path)
    
  def _build_model(self):
    """Builds the computational graph for our model."""
    
    # 1. Let's create a (source) word embeddings matrix. 
    # These are trainable parameters, so we use tf.get_variable.
    # Shape: [Vx, emb_dim] where Vx is the source vocabulary size
    x_embeddings = tf.get_variable(
      name="x_embeddings", initializer=tf.random_normal_initializer(),
      shape=[self.x_vocabulary_size, self.emb_dim])    
    
    # Now we start defining our graph.
    
    # This looks up the embedding vector for each word given the word IDs in self.x.
    # Shape: [B, M, emb_dim] where B is batch size, M is (longest) source sentence length.
    x_embedded = tf.nn.embedding_lookup(x_embeddings, self.x)

    # 2. Now we define the generative model P(Y | X=x)
    
    # first we need to know some sizes from the current input data
    batch_size = tf.shape(self.x)[0]
    longest_x = tf.shape(self.x)[1]  # longest M
    longest_y = tf.shape(self.y)[1]  # longest N
    
    # It's also useful to have masks that indicate what
    # values of our batch we should ignore.
    # Masks have the same shape as our inputs, and contain
    # 1.0 where there is a value, and 0.0 where there is padding.
    x_mask = tf.cast(tf.sign(self.x), tf.float32)  # Shape: [B, M]
    y_mask = tf.cast(tf.sign(self.y), tf.float32)  # Shape: [B, N]
    x_len = tf.reduce_sum(tf.sign(self.x), axis=1)  # Shape: [B]
    y_len = tf.reduce_sum(tf.sign(self.y), axis=1)  # Shape: [B]
    
    # 2.a Build an alignment model P(A | X, M, N)
    
    # This just gives you 1/length_x (already including NULL) per sample.
    # i.e. the lengths are the same for each word y_1 .. y_N.
    lengths = tf.expand_dims(x_len, -1)  # Shape: [B, 1]
    pa_x = tf.div(x_mask, tf.cast(lengths, tf.float32))  # Shape: [B, M]
    
    # We now have a matrix with 1/M values.
    # For a batch of 2 setencnes, with lengths 2 and 3:
    #
    #  pa_x = [[1/2 1/2   0]
    #          [1/3 1/3 1/3]]
    #
    # But later we will need it N times. So we repeat (=tile) this
    # matrix N times, and for that we create a new dimension
    # in between the current ones (dimension 1).
    pa_x = tf.expand_dims(pa_x, 1)  # Shape: [B, 1, M]
    
    #  pa_x = [[[1/2 1/2   0]]
    #          [[1/3 1/3 1/3]]]
    # Note the extra brackets.

    # Now we perform the tiling:
    pa_x = tf.tile(pa_x, [1, longest_y, 1])  # [B, N, M]    
    # Result:
    #  pa_x = [[[1/2 1/2   0]
    #           [1/2 1/2   0]]
    #           [[1/3 1/3 1/3]
    #           [1/3 1/3 1/3]]]
    
    # 2.b P(Y | X, A) = P(Y | X_A)
        
    # First we make the input to the MLP 2-D.
    # Every output row will be of size Vy, and after a softmax
    # will sum to 1.0. 
    mlp_input = tf.reshape(x_embedded, [batch_size * longest_x, self.emb_dim])
    
    # Here we apply the MLP to our input.
    h = tf.matmul(mlp_input, self.mlp_W_) + self.mlp_b_  # affine transformation
    h = tf.tanh(h)                                       # non-linearity  
    h = tf.matmul(h, self.mlp_W) + self.mlp_b            # affine transformation [B * M, Vy]
    
    # Now we perform a softmax which operates on a per-row basis.
    py_xa = tf.nn.softmax(h)
    py_xa = tf.reshape(py_xa, [batch_size, longest_x, self.y_vocabulary_size])
    
    # 2.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)

    # Here comes a rather fancy matrix multiplication.
    # Note that tf.matmul is defined to do a matrix multiplication
    # [N, M] @ [M, Vy] for each item in the first dimension B.
    # So in the final result we have B matrices [N, Vy], i.e. [B, N, Vy].
    #
    # We matrix-multiply:
    #   pa_x       Shape: [B, N, *M*]
    # and
    #   py_xa      Shape: [B, *M*, Vy]
    # to get 
    #   py_x  Shape: [B, N, Vy]
    #
    # Note: P(y|x) = prod_j p(y_j|x) = prod_j sum_aj p(aj|m)p(y_j|x_aj) 
    #
    py_x = tf.matmul(pa_x, py_xa)  # Shape: [B, N, Vy]

    # This calculates the accuracy, i.e. how many predictions we got right.
    predictions = tf.argmax(py_x, axis=2)
    acc = tf.equal(predictions, self.y)
    acc = tf.cast(acc, tf.float32) * y_mask
    acc_correct = tf.reduce_sum(acc)
    acc_total = tf.reduce_sum(y_mask)
    acc = acc_correct / acc_total
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=self.y,
      logits=tf.log(py_x),
      name="logits"
    )
    cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)
    cross_entropy = tf.reduce_mean(cross_entropy)
    
    # Now we define our cross entropy loss   
#     y_one_hot = tf.one_hot(self.y, depth=self.y_vocabulary_size)     # [B, N, Vy]
#     cross_entropy = tf.reduce_sum(y_one_hot * tf.log(py_x), axis=2)  # [B, N]
#     cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)    # [B]
#     cross_entropy = -tf.reduce_mean(cross_entropy)  # scalar

    self.pa_x = pa_x
    self.py_x = py_x  
    self.py_xa = py_xa
    self.loss = cross_entropy
    self.predictions = predictions
    self.accuracy = acc
    self.accuracy_correct = tf.cast(acc_correct, tf.int64)
    self.accuracy_total = tf.cast(acc_total, tf.int64)
    
    
  def evaluate(self, data, ref_alignments):
    """Evaluate the model on a data set."""
    
    ref_align = read_naacl_alignments(ref_alignments)
    
    align_iterator = iter(ref_align)
    metric = AERSufficientStatistics()
    accuracy_correct = 0
    accuracy_total = 0
    
    for batch_id, batch in enumerate(iterate_minibatches(data, batch_size=4)):
      
      x, y = prepare_data(batch, self.x_vocabulary, self.y_vocabulary)
      y_len = np.sum(np.sign(y), axis=1, dtype="int64")
      
      align, prob, acc_correct, acc_total = self.get_viterbi(x, y) 
      accuracy_correct += acc_correct
      accuracy_total += acc_total
      
#       if batch_id == 0:
#         print(batch[0])      
      
      for alignment, N, (sure, probable) in zip(align, y_len, ref_align):
        # the evaluation ignores NULL links, so we discard them
        # j is 1-based in the naacl format
        pred = set((aj, j) for j, aj in enumerate(alignment[:N], 1) if aj > 0)
        metric.update(sure=sure, probable=probable, predicted=pred)

    accuracy = accuracy_correct / float(accuracy_total)
    return metric.aer(), accuracy

    
  def get_viterbi(self, x, y):
    """Returns the Viterbi alignment for (x, y)"""
    
    feed_dict = {
      self.x : x,  # English
      self.y : y   # French
    }
    
    # run model on this input
    py_xa, acc_correct, acc_total = self.session.run(
      [self.py_xa, self.accuracy_correct, self.accuracy_total], 
      feed_dict=feed_dict)
    
    # things to return
    batch_size, longest_y = y.shape
    alignments = np.zeros((batch_size, longest_y), dtype="int64")
    probabilities = np.zeros((batch_size, longest_y), dtype="float32")
    
    for b, sentence in enumerate(y):
      for j, french_word in enumerate(sentence):
        if french_word == 0:  # Padding
          break
          
        probs = py_xa[b, :, y[b, j]]
        a_j = probs.argmax()
        p_j = probs[a_j]
        
        alignments[b, j] = a_j
        probabilities[b, j] = p_j
    
    return alignments, probabilities, acc_correct, acc_total

  
