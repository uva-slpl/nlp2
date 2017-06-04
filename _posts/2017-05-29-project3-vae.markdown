---
layout: post
title:  Project 3 VAE
date:   2017-05-29
author: Joost
categories: projects
---

Hi everyone,

We've created a Variational Autoencoder with Categorical loss. We hope that this helps you.

So, the resources for project 3 are now the following:

1. [Theory notebook](https://github.com/uva-slpl/nlp2/blob/gh-pages/resources/project_neuralibm/theory.ipynb) 
2. [Neural IBM1](https://github.com/uva-slpl/nlp2/blob/gh-pages/resources/project_neuralibm/neural-ibm1.ipynb)
3. [VAE with categorical loss](https://github.com/uva-slpl/nlp2/blob/gh-pages/resources/project_neuralibm/vae.ipynb) 


Lastly, if you don't have much experience with TensorFlow yet, you might want to take a look at [TF mechanics](https://www.tensorflow.org/get_started/mnist/mechanics).



Best,

Wilker & Joost


# FAQ

* Where are the manual alignments for the test set?

We have uploaded the alignments, you should be able to find them on github now.

* How do I do model selection?

Just like you did for project 1. That is, you should use *validation* log-likelihood and *validation* AER to track the performance after each epoch. At the end of training, you assess performance on *test set* using the best parameters you've got.

* How do I track log-likelihood of training data?

That's tricky. You don't really want to track the likelihood of the complete training data because that would be rather expensive. One strategy is to plot likelihood of each training mini-batch. Another strategy is to keep a running average for each epoch (that's what Joost implemented for you). 



