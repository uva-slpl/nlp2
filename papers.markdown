---
layout: photolist
title: Papers
---

{% assign hashes = (site.data.papers) %}
{% capture years %}
{% for hash in hashes %}
{{ hash[0] }}
{% endfor %}
{% endcapture %}

{% assign sortedyears = years | split:' ' | sort | reverse %}
{% for year in sortedyears %}
### {{ year }}
{% for paper in hashes[year] %}
{% include paper.html paper=paper %}
{% endfor %}
{% endfor %}
