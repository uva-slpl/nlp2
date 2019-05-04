---
layout: default
title: Projects
menu: yes
---

*Spring 2019*

We want you to get your hands dirty with most of the core topics covered in the course. 
To that end, we prepared two projects. 

Groups: check Canvas or our blog posts.

# Project 1 

*Deadline: April 26th, 2019*



In this project you will learn about lexical alignment, the task of learning correspondences between words in different languages.
You will apply latent variable modelling techniques, in particular, learning with directed graphical models (aka locally normalised models).
In this project, you will parameterise the model using categorical distributions. 
You will experiment with maximum likelihood estimation.

* Maximum likelihood estimation for IBM1 and IBM2: EM algorithm

**Note: in IBM2 experiment use relative jumps (jump function).**


Resources:

* [Project description](resources/project_ibm/project1.pdf)
* [Training data](resources/project_ibm/training.tgz)
* [Validation data](resources/project_ibm/validation.tgz)
* [Test data](resources/project_ibm/testing.tgz)  ``new!``
* [Neural IBM1](resources/project_ibm/neuralibm.tar.gz)  ``EXTRA!``
<!---* [Tips](https://uva-slpl.github.io/nlp2/projects/2018/04/12/project1.html)--->
* [Helper functions for validation AER](resources/project_ibm/aer.py)
    * Note that I wrote this helper class using `python3`, if you are using `python2` you will need to import `from __future__ import division` to avoid the default integer division (whereby stuff like 1/2 evaluates to 0 instead of 1.5)

Submission:

TBA

Assessment: [guidelines](resources/project_ibm/assessment-sheet.pdf) /  grades on Canvas.


# Project 2 

*Deadline: May 20th, 2019*

In this project you will learn and implement a deep generative language model. 
Resources:

* [Project description](resources/project_senvae/Project_2__Sentence_VAE.pdf)
* [Training data](resources/project_senvae/data/02-21.10way.clean)
* [Validation data](resources/project_senvae/data/22.auto.clean)
* [Test data](resources/project_senvae/data/23.auto.clean)

Assessment: described in the project description / grades on Canvas.
