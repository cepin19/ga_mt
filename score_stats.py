import sys
scores=[]
for line in sys.stdin:
    scores.append(float(line.split(":")[1].strip()))
print(" min:{} avg: {} max: {}".format(min(scores),sum(scores)/len(scores),max(scores)))
try:
	import plotille
	print(plotille.hist(scores))
except ImportError:
	print("No plotille")

