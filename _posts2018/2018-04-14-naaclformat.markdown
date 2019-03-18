---
layout: post
title:  NAACL format
date:   2018-04-14
author: Miguel
categories: projects
---


NAACL alignment format
----------------------

Output format
-------------
The results file should include one line for each word-to-word alignment 
identified by the system. The lines in the results file should follow the 
format below:     

sentence_no position_L1 position_L2 [S|P] 

where:
- sentence_no represents the id of the sentence within the test file. 
Sentences in the test data already have an id assigned. (see the examples 
below)    
- position_L1 represents the position of the token that is aligned from 
the text in language L1; the first token in each sentence is token 1. (not 0)    
- position_L2 represents the position of the token that is aligned from the 
text in language L2; again, the first token is token 1.    
- S|P can be either S or P, representing a Sure or Probable alignment. All 
alignments that are tagged as S are also considered to be part of the P 
alignments set (that is, all alignments that are considered "Sure" alignments 
are also part of the "Probable" alignments set). If the S|P field is missing, 
a value of S will be assumed by default.
The S|P field overlap is optional. 

Running example
---------------
Consider the two following aligned sentences:

[from the English file]

They had gone . 

[from the French file]

Ils etaient alles .

A correct word alignment that will be produced for this sentence is

18 1 1

18 2 2 

18 3 3

18 4 4

Which states that all these alignments are from sentence 18, and the English 
token 1 ("They") aligns with the French token 1 ("Ils"), the English token 2 
("had"), aligns with the French token 2 ("etaient"), and so on. Note that the 
punctuation is also aligned (English token 4 (".") align with French token 
(".")), and will count towards the final scoring figures. 
With missing S|P fields considered by default to be S. 
