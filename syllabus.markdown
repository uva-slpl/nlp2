---
layout: photolist
title: Syllabus
menu: yes
---

*Spring 2018* &#124; [Spring 2017](2017)

In this course you will apply several machine learning techniques to deal with structure predictions problems related to translation.
Our course is composed of 3 blocks of lectures: lexical alignment (3 lectures), statistical machine translation (4 lectures), and neural machine translation (4 lectures).
We will also have lab sessions related to project assignments.

# Lectures

{% assign lectures = (site.data.2018.intro | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}

## Lexical alignment

{% assign lectures = (site.data.2018.alignment | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}

## Statistical machine translation

{% assign lectures = (site.data.2018.smt | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}

## Neural machine translation

{% assign lectures = (site.data.2018.nmt | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}



