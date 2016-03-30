---
layout: photolist
title: Instructors
---

[Statistical Language Processing and Learning Lab][SLPLL] (minus just a few).

{% assign instructors = (site.data.instructors | where: "selected", "y") %}
{% for person in instructors %}
{% include person.html person=person %}
{% endfor %}



[SLPLL]: {{ site.slpll_url }} "Statistical Language Processing and Learning Lab"
