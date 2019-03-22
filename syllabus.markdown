---
layout: photolist
title: Syllabus
menu: yes
---

*Spring 2019* &#124; [Spring 2018](2018)

In this course you will apply several machine learning techniques to deal with structure predictions problems related to Natural Language Processing (NLP).
Our course is composed of 2 blocks of lectures: lexical alignment,and deep generative models for NLP.
We will also have lab sessions related to project assignments.

# Lectures

{% assign lectures = (site.data.2019.intro | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}

## Lexical alignment

{% assign lectures = (site.data.2019.alignment | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}

## Deep generative models for NLP

{% assign lectures = (site.data.2019.vae | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}




