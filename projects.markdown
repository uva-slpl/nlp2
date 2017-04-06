---
layout: default
title: Projects
---

*Spring 2017*

We want you to get your hands dirty with most of the core topics covered in the course. 
To that end, we prepared three projects.

# Project 1 

*From March 12 to March 26*

In this project you will learn about lexical alignment, the task of learning correspondences between words in different languages.
You will apply latent variable modelling techniques, in particular, learning with directed graphical models (aka locally normalised models).
In this project, you will parameterise the model using categorical distributions. 
You will experiment with maximum likelihood estimation and Bayesian modelling with Dirichlet priors.

* Maximum likelihood estimation for IBM1: EM algorithm
* Bayesian estimation for IBM1: variational Bayes

Resources:

* [Project description](resources/project_ibm/project1.pdf)
* [Training data](resources/project_ibm/training.tgz)
* [Validation data](resources/project_ibm/validation.tgz)
* [Helper functions for validation AER](resources/project_ibm/aer.py)


# Project 2 

*From March 26 to May 17*


In this project, we will focus on the problem of learning how to permute a sequence of words into target language word order, we will frame this truly unsupervised learning problem as a supervised one by relying on word alignments as a source of observations. We will use pre-trained alignments to extract a cannonical tree-structured mapping between the source word order and the target word order. 
By framing it as a supervised learning problem we can investigate a different class of probabilistic graphical models, namely, undirected models (aka globally normalised models). 

* Maximum likelihood estimation for a CRF parser: CKY algorithm, inside-outside algorithm, and gradient-based optimisation

# Project 3

*From May 17 to June 7*

In this project you will learn about maximum likelihood estimation for graphical models parameterised by neural networks.
You will investigate an unsupervised problem for which a tractable solutions exists (an IBM1-type model) and an unsupervised problems for which approximate inference is necessary (an embed-and-align type of model).

* IBM1 with NNs: a NN predicts the parameters of the lexical categorical distributions (note that this requires to explicitly marginalise over alignments);
* IBM1 without null words: this model adds a latent variable that can propose segmentations trading between translation and monolingual insertion (again this requires explicit marginalisation of translation/insertion decisions);
* Jointly learn how to embed and align: along with alignment and segmentation, learn latent random embeddings (where marginalisation of latent embeddings is intractable in general).

In this project you will employ techniques like explicit marginalisation of latent alignments and variational inference to circumvent the explicit marginalisation of latent embeddings.


