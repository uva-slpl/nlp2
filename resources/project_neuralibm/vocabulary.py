#!/usr/bin/env python3

from collections import Counter, OrderedDict
import numpy as np


class OrderedCounter(Counter, OrderedDict):
  """A Counter that remembers the order in which items were added."""
  pass
  

class Vocabulary:
  """A simple vocabulary class to map words to IDs."""
  
  def __init__(self, corpus=None, 
               special_tokens=('<PAD>', '<UNK>', '<S>', '</S>', '<NULL>'), max_tokens=0):
    """Initialize and optionally add tokens in corpus."""

    self.counter = OrderedCounter()
    self.t2i = OrderedDict()
    self.i2t = []
    self.special_tokens = [t for t in special_tokens]
            
    if corpus is not None:
      for tokens in corpus:
        self.counter.update(tokens)
        
    if max_tokens > 0:
      self.trim(max_tokens)
    else:  
      self.update_dicts()
 
  def __contains__(self, token):
    """Checks if a token is in the vocabulary."""
    return token in self.counter
  
  def __len__(self):
    """Returns number of items in vocabulary."""
    return len(self.t2i)
  
  def get_token_id(self, token):
    """Returns the ID for token, if we know it, otherwise the ID for <UNK>."""
    if token in self.t2i:
      return self.t2i[token]
    else:
      return self.t2i['<UNK>']
    
  def tokens2ids(self, tokens):
    """Converts a sequence of tokens to a sequence of IDs."""
    return [self.get_token_id(t) for t in tokens]
    
  def get_token(self, i):
    """Returns the token for ID i."""
    if i < len(self.i2t):
      return self.i2t[i]
    else:
      raise IndexError("We do not have a token with that ID!")

  def add_token(self, token):
    """Add a single token."""
    self.counter.add(token)
    
  def add_tokens(self, tokens):
    """Add a list of tokens."""
    self.counter.update(tokens)   
    
  def update_dicts(self):
    """After adding tokens or trimming, this updates the dictionaries."""
    self.t2i = OrderedDict()
    self.i2t = list(self.special_tokens)
    
    # add special tokens
    self.i2t = [t for t in self.special_tokens]
    for i, token in enumerate(self.special_tokens):
      self.t2i[token] = i
    
    # add tokens
    for i, token in enumerate(self.counter, len(self.special_tokens)):
      self.t2i[token] = i
      self.i2t.append(token) 
      
  def trim(self, max_tokens):
    """
    Trim the vocabulary based on frequency. 
    WARNING: This changes all token IDs.
    """
    tokens_to_keep = self.counter.most_common(max_tokens)
    self.counter = OrderedCounter(OrderedDict(tokens_to_keep))
    self.update_dicts()
    
  def batch2tensor(self, batch, add_null=True, add_end_symbol=True):
    """
    Returns a tensor (to be fed to a TensorFlow placeholder) from a batch of sentences.
    The batch input is assumed to consist of tokens, not IDs. 
    They will be converted to IDs inside this function.
    """
    # first we find out the shape of the tensor we return
    batch_size = len(batch)
    max_timesteps = max([len(x) for x in batch])

    if add_end_symbol:
      max_timesteps += 1
    
    if add_null:
      max_timesteps += 1    
    
    # then we create an empty tensor, consisting of zeros everywhere
    tensor = np.zeros([batch_size, max_timesteps], dtype='int64')
    
    # now we fill the tensor with the sequences of IDs for each sentence
    for i, sequence in enumerate(batch):
      
      start = 1 if add_null else 0
      tensor[i, start:len(sequence)+start] = self.tokens2ids(sequence)
      
      if add_null:
        tensor[i, 0] = self.get_token_id("<NULL>")
      
      if add_end_symbol:
        tensor[i, -1] = self.get_token_id("</S>")  # end symbol
      
    return tensor   


