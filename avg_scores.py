import sys
if len(sys.argv)==2:
    i=int(sys.argv[1])
else:
    i=-1

lines=[float(line) for line in sys.stdin][:i]
print(sum(lines)/len(lines))


