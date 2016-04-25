---
layout: post
title:  Syllabus
date:   2016-03-29
categories: info
---

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  
<!-- *generated with [DocToc](https://github.com/thlorenz/doctoc)* -->

- [Introduction (29 March)](#introduction-29-march)
- [Word alignment models (31 March)](#word-alignment-models-31-march)
- [ITG-based models (05 April)](#itg-based-models-05-april)
- [Phrase-based models (07 April)](#phrase-based-models-07-april)
- [Hierarchical models (12 April)](#hierarchical-models-12-april)
- [Evaluation and tuning (14 April)](#evaluation-and-tuning-14-april)
- [Word-order and reordering grammar (21 April)](#word-order-and-reordering-grammar-21-april)
- [Labelling Hiero (26 April)](#labelling-hiero-26-april)
- [Translating into morphologically rich languages (28 April)](#translating-into-morphologically-rich-languages-28-april)
- [Neural models for translation (03 May)](#neural-models-for-translation-03-may)
- [Multimodal MT (10 May)](#multimodal-mt-10-may)
- [Parallel text as a linguistic resource (12 May)](#parallel-text-as-a-linguistic-resource-12-may)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Introduction (29 March)

1. Introduction to the course
2. Schedule


# Word alignment models (31 March)

1. Aligning by learning a translation lexicon
2. Estimation by MLE: expectation maximisation
3. The role of word order

**Reading**

* [Lecture notes](resources/papers/CollinsIBM.pdf)

# ITG-based models (05 April)

1. Word order: constraining the space of permutations
2. Inversion transduction grammar
3. Estimation
4. Decoding

**Reading**

* [Stochastic inversion transduction grammars and bilingual parsing of corpora](https://www.aclweb.org/anthology/J/J97/J97-3002.pdf)

# Phrase-based models (07 April)

1. A larger translation unit
2. Phrase extraction from word alignments
3. Reordering of phrases
4. Generation

**Reading**

* [Statistical phrase-based translation](http://www.aclweb.org/anthology/N/N03/N03-1017.pdf)

**Discussion**

* [A phrase-based, joint probability model for SMT](http://www.aclweb.org/anthology/W02-1018)

# Hierarchical models (12 April)

1. Lexicalised ITG rules: evidence for reordering
2. Synchronous context-free grammars
3. Rule extraction from word alignments
4. Generation

**Reading**

* [Chiang's SCFG Tutorial]()

**Discussion**

* [A hierarchical phrase-based model for SMT](http://www.aclweb.org/anthology/P05-1033)

# Evaluation and tuning (14 April)

1. Automatic MT evaluation
2. Tuning 

**Reading**

**Discussion**

# Word-order and reordering grammar (21 April)

1. Preordering for machine translation
2. Factorizing permutations 
3. Reordering grammar

**Reading**

* [Reordering grammar induction](http://www.aclweb.org/anthology/D15-1005)

**Discussion**

* [Learning linear ordering problems for better translation](http://www.aclweb.org/anthology/D09-1105)

# Labelling Hiero (26 April)

1. Syntactic labelling: disambiguating hierarchical rules
2. Labelling by factorizing word alignments

**Reading**
* [Hierarchical alignment decomposition labels for Hiero grammar rules](http://www.aclweb.org/anthology/W13-0803)

**Discussion**

* [Learning hierarchical translation structure with linguistic annotation](http://www.aclweb.org/anthology/P11-1065)

# Translating into morphologically rich languages (28 April)

1. Morphology and word-order
2. Transferring morphology through word alignments
3. Predicting morphology and word-order

# Neural models for translation (03 May)

1. Language model
2. Encode/decoder
3. NMT

# Multimodal MT (10 May)

1. Image representations
2. Text representations
3. Joint representations: potential


# Parallel text as a linguistic resource (12 May)

1. Transfer learning
2. Crosslingual applications
    * Disambiguation
    * Tagging
    * Morphology
