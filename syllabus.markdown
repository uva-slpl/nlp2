---
layout: photolist
title: Syllabus
menu: yes
---

*Spring 2019* &#124; [Spring 2018](2018)

In this course you will apply several machine learning techniques to deal with structure predictions problems related to translation.
Our course is composed of 2 blocks of lectures: lexical alignment,and neural machine translation.
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

## Neural machine translation

{% assign lectures = (site.data.2019.nmt | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}




