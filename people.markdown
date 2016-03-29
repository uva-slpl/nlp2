---
layout: photolist
title: People
---

# Instructors

{% assign instructors = (site.data.people.instructors | where: "selected", "y") %}
{% for person in instructors %}
{% include person.html person=person %}
{% endfor %}


