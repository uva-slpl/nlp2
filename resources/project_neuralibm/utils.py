import gzip


def smart_reader(path, encoding='utf-8', delimiter=' '):
  """Read in a gzipped file and return line by line"""
  if path.endswith(".gz"):
    with gzip.open(path, mode='r') as f:
      for line in f:
        yield str(line, encoding=encoding).strip().split(delimiter)
  else:
    with open(path, encoding=encoding) as f:
      for line in f:
          yield line.strip().split(delimiter)    


def bitext_reader(src_sequences, trg_sequences, max_length=0):
  """
  Reads in a parallel corpus (bitext) and returns tokenzied sentence pairs.
  Note: it's possible to implement some more data filtering here.
  """
  for src_seq in src_sequences:
    trg_seq = next(trg_sequences)
    
    # filter
    if max_length > 0:
      if len(src_seq) > max_length or len(trg_seq) > max_length:
        continue
      
    yield src_seq, trg_seq


def iterate_minibatches(corpus, batch_size=16):
  """Return a mini-batch at a time from corpus."""
  batch = []
  for sequence in corpus:
    batch.append(sequence)
    if len(batch) == batch_size:
      yield batch
      batch = []

      
def prepare_data(batch, vocabulary_x, vocabulary_y):
  """Prepare batch of sentences for TensorFlow input."""
  batch_x, batch_y = zip(*batch)
  x = vocabulary_x.batch2tensor(batch_x, add_null=True, add_end_symbol=False)      
  y = vocabulary_y.batch2tensor(batch_y, add_null=False, add_end_symbol=False)    
  return x, y

def prepare_batch_data(batch, vocabulary):
  """Prepare batch of sentences for TensorFlow input."""
  x = vocabulary.batch2tensor(batch, add_null=True, add_end_symbol=False)
  return x


def filter_len(data, max_length=30):
  for x in data:
    if len(x) > max_length:
      continue
    yield x

