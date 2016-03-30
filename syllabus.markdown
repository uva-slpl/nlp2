---
layout: photolist
title: Syllabus
---


{% assign lectures = (site.data.lectures | where: "selected", "y") %}
{% for lecture in lectures %}
{% include lecture.html lecture=lecture %}
{% endfor %}


