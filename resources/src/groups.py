import csv
from collections import defaultdict

reader = csv.reader(open('teams2017.csv'))
groups = defaultdict(list)

header = next(reader)

for row in reader:
    gid = row[0]  # group id
    sid = row[2]  # student id
    name = row[-2] + ' ' + row[-1]  # full name
    groups[gid].append((sid, name))

for i, (gid, members) in enumerate(sorted(groups.items(), key=lambda pair: pair[0]), 1):
    print('* Group %d: %s' % (i, ', '.join('%s (%s)' % (name, sid) for sid, name in members)))

