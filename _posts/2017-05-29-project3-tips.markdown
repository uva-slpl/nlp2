---
layout: post
title:  Project 3 - Tips
date:   2017-06-04
author: Wilker
categories: projects
---

# Tips

* Where are the manual alignments for the test set?

We have uploaded the alignments, you should be able to find them on github now.

* How do I do model selection?

Just like you did for project 1. That is, you should use *validation* log-likelihood and *validation* AER to track the performance after each epoch. At the end of training, you assess performance on *test set* using the best parameters you've got.

* How do I track log-likelihood of training data?

That's tricky. You don't really want to track the likelihood of the complete training data because that would be rather expensive. One strategy is to plot likelihood of each training mini-batch. Another strategy is to keep a running average for each epoch (that's what Joost implemented for you). 

* Are the models in T2 better than the basic neural IBM 1?

Not really. Sometimes with enough tricks (e.g. like getting the right layer size, right number of nonlinear layers, some regularisation) you can get it to be as good as the basic model or slightly better. But that's what I call *death by hyperparameter optimisation* and we do not find that exciting. The most common case is that T2 will underperform. 

* Are T3/T4 better than the basic neural IBM 1?

In my experience yes, about 2 AER, but I only tried in a larger scale scenario.

* How many terms do I need in the Taylor expansion of the KL?

Use something like 5 to 10.

* What if I get numerical problems when computing the KL?

You can constrain your predictions for parameters of Beta and Kuma to be in (0, k] where k is some small positive integer (e.g. 10). For example you can clip your activations to that interval. This is not super satisfactory, but better strategies are considerably more involved. 

* Do I need NULL words on the English side in T3 and T4?

Actually, you don't! The collocation variables mediate between translation and insertion which makes NULL words unnecessary. 

* What if I already did everything with NULL words in T3 and T4?

That's okay. Now you know you didn't have to ;)


