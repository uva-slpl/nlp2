---
layout: default
title: Projects
---

*Spring 2017*

We want you to get your hands dirty with most of the core topics covered in the course. 
To that end, we prepared three projects. 

Groups: check Blackboard or our blog posts.


# Project 1 

*From April 12 to April 26*

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
* [Test data](resources/project_ibm/testing.tgz)  ``new!``
* [Tips](https://uva-slpl.github.io/nlp2/project1/2017/04/12/IBM.html)
* [Helper functions for validation AER](resources/project_ibm/aer.py)
    * Note that I wrote this helper class using `python3`, if you are using `python2` you will need to import `from __future__ import division` to avoid the default integer division (whereby stuff like 1/2 evaluates to 0 instead of 1.5)

Submission:

* Upload tgz file on Blackboard before *April 29*, 10:00. (Note: deadline extended!)


# Project 2 

In this project we will frame translation as a latent variable model CRF.
We will employ ITGs to constrain the space of translation derivations and experiment with maximum likelihood estimation for CRFs.

Topics:

* Algorithms: Earley intersection, Topsort, Inside-Outside, Viterbi, and ancestral sampling
* Undirected graphical models: CRF
* MLE by gradient-based optimisation

Resources:

* Make sure to check our [blogpost](https://uva-slpl.github.io/nlp2/projects/2017/05/03/project2.html) `new!`
* [Project description](resources/project_crf/project2.pdf)
* [Notes on Earley intersection](resources/papers/Aziz-Earley.pdf)
* [Data](resources/project_crf/data.tgz)
* [Dev target lengths](resources/project_crf/dev123_lengths.tgz) `new!`
* [Lexicon](resources/project_crf/lexicon.tgz) Format: `Chinese English P(en|zh) P(zh|en)`
* [Notebooks](https://github.com/uva-slpl/nlp2/tree/gh-pages/resources/notebooks)
    * check `libitg` which contains the complete parser
    * `LV-CRF-Roadmap` is a notebook that uses `libitg` and discusses the steps towards completing project 2

Submission:

* Upload .tgz file on Blackboard before *Wednesday May 24*, 23:59 (GMT-8) `new!`

# Project 3

*From May 17 to June 7*

In this project you will learn about maximum likelihood estimation for graphical models parameterised by neural networks.
You will investigate an unsupervised problem for which a tractable solutions exists (an IBM1-type model) and an unsupervised problems for which approximate inference is necessary (an embed-and-align type of model).

* IBM1 with NNs: a NN predicts the parameters of the lexical categorical distributions (note that this requires to explicitly marginalise over alignments);
* IBM1 without null words: this model adds a latent variable that can propose segmentations trading between translation and monolingual insertion (again this requires explicit marginalisation of translation/insertion decisions);
* Jointly learn how to embed and align: along with alignment and segmentation, learn latent random embeddings (where marginalisation of latent embeddings is intractable in general).

In this project you will employ techniques like explicit marginalisation of latent alignments and variational inference to circumvent the explicit marginalisation of latent embeddings.

