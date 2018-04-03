---
layout: default
title: Assessment
menu: no
---

Here you will find the criteria used in assessing your reports. We might also post some tips you can consult when preparing your report.

# Project 1

Your conference-style report will be assessed by two independent reviewers according to the following evaluation criteria:

* Scope (max 2 points): *Is the problem well presented? Do students understand the challenges/contributions?* Here we expected to learn about:
    * the alignment problem
    * the problem with maximum-likelihood estimation, which motivates a Bayesian extension
    * the problem with posterior inference, which motivates the use of variational inference
* Theoretical description (max 3 points): *Are the models presented clearly and cor- rectly?* Here we expected to learn about:
    * model formulation: assumptions, factorisation, parameterisation, limitations
    * parameter estimation
    * inference techniques
* Empirical evaluation (max 3 points): *Is the experimental setup sound/convincing? Are experimental findings presented in an organised and effective manner?* Here we expected to learn about:
    * the data
    * the experimental setup
    * training conditions (the various choices and hyperparameters discussed and justified)
    * ablation studies if applicable
    * test results
    * a critical discussion of findings
* Writing style (max 2 points): *use of latex, structure of report, use of tables/figures/plots, command of English.* Here we expected
    * good use of latex and compliance with the format specified in the project description (w.r.t. template and length)
    * clear structure
    * good use of visualization techniques (e.g. tables, figures, and plots)
    * command of English
* Extra (max 1 point): *variational Bayes for IBM model 2 (at least for the lexical model)*


# Project 2

Your conference-style report will be assessed by two independent reviewers according to the following evaluation criteria:

* Scope (max 2 points): 
* Theoretical description (max 3 points):
* Empirical evaluation (max 3 points): 
* Writing style (max 2 points): 
* Extra (max 1 point): 




Scope:

* Introduce the problem: translation along with unsupervised induction of a hierarchical mapping between two strings.
* Discuss the challenge with permutations and what ITGs give you.
* Discuss the benefits of an undirected model. Contrast it with directed models. 

Theoretical background:

* Clearly present graphical model and assumptions.
* Clearly present the grammar and constraints you are using.
* No pseudo codes: I'm pretty familiar with graph algorithms. Instead, make sure it's obvious from your report that you understand the role of each algorithm in computing some key quantity. Discuss the quantity, the role of algorithm, refer to literature.
* Present your learning technique: MLE via gradient based optimisation. Derive the objective clearly. Discuss parameter updates and regularisation techniques.
* Discuss prediction techniques: assumptions, approximations, caveats, etc.

Experiments:

* You have hyper-parameters: SGD learning rate schedule, regularisation strength. Investigate them. Plot likelihood and validation BLEU.
* You have features of 3 sorts: segmentation, lexical translation, word order. Conduct ablation experiments.
* Report BLEU1 (it gives an ideal of how well your model performs lexical selection), report BLEU4 (it gives an idea of lexical selection and word ord).

Style:

* Latex! 
* Use plots, tables, and figures, where appropriate.
* Use the ACL template.
* Proper bibtex.
* Cite papers, not slides.
* Write a research paper, not a technical report.
* Be critical.
