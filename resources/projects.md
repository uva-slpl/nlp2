
# Fully supervised neural transducer

This is basically a simpler machine translation model, where training instances are pre-ordered and segmented such that one can see the data as a stream of monotone phrase pairs. The project involves learning a simple transducer that can deal with variable length source phrases and variable length target phrases, but translate these monotonically. Here we recommend one way to approach the problem (with a pipeline):

1. Learn a classifier that predicts the length of a target phrase given a source phrase
2. Learn a monotone transducer that i) encodes source phrases through a fixed or learned average of word embeddings, and ii) decodes word by word with a feedforward neural network over a fixed target history and a context vector representing the phrase we are currently translating. 

Note that in step (2) we know how many target words we have to generate before moving on to the next source phrase because of step (1).

# Unsupervised alignment, segmentation and embedding

You will investigate an unsupervised problem for which a tractable solutions exists (an IBM1-type model) and an unsupervised problems for which approximate inference is necessary (an embed-and-align type of model).

* IBM1 with NNs: a NN predicts the parameters of the lexical categorical distributions
* IBM1 without null words: this model adds a latent variable that can propose segmentations (trading between translation and monolingual insertion)
* Jointly learn how to embed and align: along with (discrete) alignments, learn latent random embeddings.

In this project you will employ techniques like explicit marginalisation of latent alignments and variational inference to circumvent the explicit marginalisation of latent embeddings.


