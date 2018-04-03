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

You can constrain your predictions for parameters of Beta and Kuma to be in [0.001, k] where k is some small positive integer (e.g. 10). For example you can clip your exponential activations to that interval or you can scale a sigmoid activation by k (and sum 0.001 to stay far enough from 0). 

After doing so, you will still observe some negative KL, the reason is because when the Kuma and the Beta overlap considerably KL should be 0. The truncated Taylor expansion gets worse as your Kuma/Beta parameters get close to 0 or too large. You will notice that when the parameters are very similar to each other, KL is going to be some small negative number. But you should notice that they are really small in magnitude (that means, still close to zero).

The problem is that a negative number being added to the loss becomes an opportunity for the optimiser to exploit numerical instability in order to make the loss artificially small. You can circumvent that by clipping the KL term from below at 0.

I know this little tweaks are not super satisfactory, but better strategies are considerably more involved. 
I added an example to the repository, check it in case you are struggling to get the KL term right!


* Do I need NULL words on the English side in T3 and T4?

Actually, you don't! The collocation variables mediate between translation and insertion which makes NULL words unnecessary. 

* What if I already did everything with NULL words in T3 and T4?

That's okay. Now you know you didn't have to ;)

* How do I decode in the collocation model?

I would suggest you simply argmax over both latent variables, for each French position. Then if you get a collocation variable to be set to 1, you can consider that word NULL-aligned.


